import torch
import numpy as np
from .components import InterleavedFSDPTrainer, BatchedAlternatingDataset
from transformers import TrainingArguments
import torchaudio.transforms as T
from collections import defaultdict
from datasets import load_dataset, Dataset
import whisper
whisper_model = whisper.load_model("small")


class AudioChatDataCollator:
    def __init__(self, tokenizer, whisper_model, model):
        self.tokenizer = tokenizer
        self.whisper_model = whisper_model
        self.model = model
        pass

    def _process_audio_tensor(audio, sample_rate=16000):
        audio = audio.to(torch.float32)
        duration_ms = (len(audio) / sample_rate) * 1000
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return mel, int(duration_ms / 20) + 1

    def _inference_collator(self, audio_input, user_res, ass_res):
        user_input_ids = self.tokenizer(
            user_res, return_tensors="pt").input_ids
        assistant_input_ids = self.tokenizer(
            ass_res, return_tensors="pt").input_ids

        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor(
            [[128009, 128260, 128261]], dtype=torch.int64)
        final_tokens = torch.tensor([[128009]], dtype=torch.int64)

        user_tokens = torch.cat(
            [start_token, user_input_ids, end_tokens], dim=1)

        labels = torch.cat([start_token, user_input_ids, end_tokens,
                            assistant_input_ids, final_tokens], dim=1)

        true_labels = torch.full_like(labels, -100)
        true_labels[:, user_tokens.shape[1]:] = labels[:, user_tokens.shape[1]:]

        attention_mask = torch.ones_like(labels)

        audio_input = audio_input.squeeze(0)
        mel, length = self._process_audio_tensor(audio_input)
        mel = mel.to(whisper_model.device)
        mel = mel.unsqueeze(0)
        audio_feature = whisper_model.embed_audio(mel)[0][:length]
        audio_feature = audio_feature.unsqueeze(0)

        return {
            "audio_values": audio_feature.to(self.model.device).to(self.model.dtype),
            "input_ids": labels.to(self.model.device),
            "labels": true_labels.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device)
        }

    def __call__(self, features):
        audio = torch.tensor([features[0]["audio"]["array"]])
        assistant_response = features[0]["assistant"]
        user_response = features[0]["user"]

        # Simple contains check
        if "<|audio|>" in user_response:
            user_response = features[0]["user"]
        else:
            user_response = "<|audio|>"

        batch = self._inference_collator(
            audio, user_response, assistant_response)

        return {
            "audio_values": batch["audio_values"].cpu(),
            "input_ids": batch["input_ids"].cpu(),
            "labels": batch["labels"].cpu(),
            "attention_mask": batch["attention_mask"].cpu()
        }


class Stage_3_Trainer():
    def __init__(
            self,
            model,
            dataset=None,
            tokenizer=None,
            save_folder="checkpoints",
            pad_token=None,
            max_length=9600,
            batch_size=None,

    ):
        self.num_threads = 1
        self.tokenizer = tokenizer
        self.model = self._prepare_model(model)
        self.max_length = max_length

        self.pad_token = pad_token

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = 8

        self.num_gpus = torch.cuda.device_count()

        self.gradient_accumulation_steps = 64//(self.num_gpus*self.batch_size)

        # some default values that can be overridden in the .train() method
        self.batch_size = 1
        self.epochs = 1
        self.save_steps = 2000
        self.learning_rate = 5.0e-6

        self.tokeniser_length = 128256
        self.start_of_text = 128000
        self.end_of_text = 128009

        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2

        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4

        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7

        self.start_of_system = self.tokeniser_length + 8
        self.end_of_system = self.tokeniser_length + 9

        self.audio_tokens_start = self.tokeniser_length + 10

        if pad_token is None:
            self.pad_token = 128263

        self.save_folder = save_folder

        # get cuda device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = dataset

        pass

    def _prepare_model(self, model):
        model = model.to(torch.bfloat16)
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "multi_modal_projector" in name:
                param.requires_grad = True
        
        return model



    def _create_training_args(self, **kwargs):

        assert self.num_gpus > 1, "At least 2 GPUs should be available for training, to allow FSDP."

        self.training_args = TrainingArguments(
            output_dir=self.save_folder,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=1,
            learning_rate=2e-3,  # Changed to 2*10^-3
            logging_steps=1,
            evaluation_strategy="no",
            report_to="wandb",
            push_to_hub=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            fp16=True,
            save_steps=15000,
            **kwargs
        )

    def _compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def create_trainer(
        self,
        **kwargs
    ):
        self._create_training_args(**kwargs)
        trainer = InterleavedFSDPTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            compute_metrics=self._compute_metrics,
            data_collator=AudioChatDataCollator(),
        )
        return trainer
