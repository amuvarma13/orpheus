from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoModel,
    PreTrainedModel,
    AutoModelForCausalLM, 
    AutoConfig
)
from dataclasses import dataclass
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput





from .components import OrpheusProjector
from .config import OrpheusConfig

@dataclass
class OrpheusCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    audio_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class OrpheusPreTrainedModel(PreTrainedModel):
    config_class = OrpheusConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class OrpheusForConditionalGeneration(OrpheusPreTrainedModel):
    def __init__(self, config: OrpheusConfig, new_vocab_size=False):
        super().__init__(config)

        self.multi_modal_projector = OrpheusProjector(config)

        self.vocab_size = config.vocab_size
        if config.text_model_id is not None:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                config.text_model_id, 
                attn_implementation=config._attn_implementation
            )
            if(new_vocab_size is not None):
                self.language_model.resize_token_embeddings(156940)
        else:
            self.language_model = AutoModelForCausalLM.from_config(
                config.text_config, 
                attn_implementation=config._attn_implementation

            )
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_audio_features(
        self, audio_features, inputs_embeds, input_ids, attention_mask, labels
    ):
        num_audio_samples, num_audio_patches, embed_dim = audio_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.pad_token_id)
        )
        special_audio_token_mask = input_ids == self.config.audio_token_index
        num_special_image_tokens = torch.sum(special_audio_token_mask, dim=-1)
        max_embed_dim = (
            num_special_image_tokens.max() * (num_audio_patches - 1)
        ) + sequence_length
        batch_indices, non_audio_indices = torch.where(
            input_ids != self.config.audio_token_index
        )

        new_token_positions = (
            torch.cumsum((special_audio_token_mask * (num_audio_patches - 1) + 1), -1)
            - 1
        )
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]

        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                self.config.ignore_index,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

        target_device = inputs_embeds.device
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)


        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_audio_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_audio_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_audio_indices
            ]

        audio_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        audio_positions = audio_to_overwrite.to(torch.int16).cumsum(-1) - 1
        audio_left_pad_mask = audio_positions >= nb_image_pad[:, None].to(target_device)
        audio_to_overwrite &= audio_left_pad_mask

        if audio_to_overwrite.sum() != audio_features.shape[:-1].numel():
            # print()
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {torch.sum(special_audio_token_mask)} while"
                f" the number of audio given to the model is {num_audio_samples}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            audio_features.contiguous()
            .reshape(-1, embed_dim)
            .to(target_device, dtype=final_embedding.dtype)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0), 1
        )

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def get_multimodal_embeddings(self, audio_values):
        speech_embeddings = self.multi_modal_projector(audio_values)
        return speech_embeddings
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        audio_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, OrpheusCausalLMOutputWithPast]:
        

   
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if (
                audio_values is not None
                and len(audio_values) > 0
                and audio_values.shape[1] > 1
                and input_ids.shape[1] != 1
            ):
                
                audio_features = self.multi_modal_projector(audio_values)
                (
                    inputs_embeds,
                    attention_mask,
                    labels,
                    position_ids,
                ) = self._merge_input_ids_with_audio_features(
                    audio_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(
                        attention_mask, self.config.ignore_index
                    ).to(torch.long)
            else:

                if (
                    past_key_values is not None
                    and audio_values is not None
                    and input_ids.shape[1] == 1
                ):

                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    batch_index, non_attended_tokens = torch.where(
                        first_layer_past_key_value.float().sum(-2) == 0
                    )

                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = torch.ones(
                        (
                            attention_mask.shape[0],
                            target_seqlen - attention_mask.shape[1],
                        ),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    valid_indices = non_attended_tokens < extended_attention_mask.size(
                        -1
                    )
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    extended_attention_mask[
                        new_batch_index, new_non_attended_tokens
                    ] = 0

                    attention_mask = torch.cat(
                        (attention_mask, extended_attention_mask), dim=1
                    )
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return OrpheusCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        audio_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]

            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            elif self.config.audio_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]

            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[
                    :, -(cache_length + input_ids.shape[1]) :
                ]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "audio_values": audio_values,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)

