from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor
import torch
import torchaudio
import whisper
from transformers import AutoModel, AutoConfig
from .model import OrpheusConfig, OrpheusForConditionalGeneration
from snac import SNAC


class OrpheusConversation():
    def __init__(self, parent):
        self.existing_embeds = None
        self.current_message = None
        self.model = parent.model
        self.special_tokens = parent.special_tokens
        self.audio_encoder = parent.audio_encoder
        self.tokenizer = parent.tokenizer
        self.parent = parent

    def append_message(self, message):
        self._validate_message(message)
        self.current_message = message

    def _validate_message(self, message):
        if "format" not in message:
            raise ValueError("Message must contain a 'format' key")
        
        if "data" not in message:
            raise ValueError("Message must contain a 'data' key")

        if message["format"] not in ["text", "speech"]:
            raise ValueError("Message format must be either 'text' or 'speech'")
        
        if message["format"] == "speech":
            if type(message["data"]) != torch.Tensor:
                raise ValueError("Speech data must be a torch.Tensor")
            if message["data"].shape[0] != 1:
                raise ValueError("Speech data must be a 1D tensor")
            
        if message["format"] == "text":
            if type(message["data"]) != str:
                raise ValueError("Text data must be a string")

    def _get_embeds(self):
        if self.current_message["format"] == "text":
            return self._get_text_embeds()
        else:
            return self._get_speech_embeds()
        
    def _get_text_embeds(self):
        print("getting text embeds")
        text = self.current_message["data"]
        text_tokens = self.tokenizer(text, return_tensors="pt").input_ids
        text_tokens = text_tokens.to(self.model.device)
        text_embeds = self.model.get_input_embeddings()(text_tokens)

        text_embeds = text_embeds.to(dtype=torch.bfloat16).to(self.model.device)
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

        start_token = start_token.to(self.model.device)
        end_tokens = end_tokens.to(self.model.device)

        start_embeds = self.model.get_input_embeddings()(start_token)
        end_embeds = self.model.get_input_embeddings()(end_tokens)

        start_embeds = start_embeds.to(dtype=torch.bfloat16)
        end_embeds = end_embeds.to(dtype=torch.bfloat16)

        if self.existing_embeds is not None:
            self.existing_embeds = self.existing_embeds.to(self.model.device).to(dtype=torch.bfloat16)
            all_embeds = torch.cat([self.existing_embeds, start_embeds, text_embeds, end_embeds], dim=1)
        else:
            all_embeds = torch.cat([start_embeds, text_embeds, end_embeds], dim=1)
        return all_embeds
    


    
    def _get_speech_embeds(self):
        audio_features = self.parent._get_audio_features(self.current_message["data"])
        audio_features = audio_features.to(dtype=torch.bfloat16).to(self.model.device)
        audio_embeds = self.model.multi_modal_projector(audio_features)
        start_token = torch.tensor([[128259, 128000]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
        start_token = start_token.to(self.model.device)
        end_tokens = end_tokens.to(self.model.device)
        start_embeds = self.model.get_input_embeddings()(start_token)
        end_embeds = self.model.get_input_embeddings()(end_tokens)
        start_embeds = start_embeds.to(dtype=torch.bfloat16)
        end_embeds = end_embeds.to(dtype=torch.bfloat16)
        if self.existing_embeds is not None:
            all_embeds = torch.cat([self.existing_embeds, start_embeds, audio_embeds, end_embeds], dim=1)
        else:
            all_embeds = torch.cat([start_embeds, audio_embeds, end_embeds], dim=1)

        return all_embeds
    
    def _update_existing_embeds(self, output_tokens):
        output_embeddings = self.model.get_input_embeddings()(output_tokens)
        end_of_ai_embedding = self.model.get_input_embeddings()(torch.tensor([[self.special_tokens["end_of_ai"]]]).to(self.model.device))
        if self.existing_embeds is None:
            self.existing_embeds = torch.cat([output_embeddings, end_of_ai_embedding], dim=1).to("cpu")
        else:
            all_embeddings = torch.cat([self.existing_embeds, output_embeddings, end_of_ai_embedding], dim=1).to("cpu")
            self.existing_embeds = all_embeddings

    
    def generate_response(self):
        if self.current_message is None:
            raise ValueError("Please append a message first")
        
        embeds = self._get_embeds()
 
        output_tokens = self.model.generate(
            inputs_embeds=embeds, 
            max_new_tokens=5000, 
            temperature=0.9,
            repetition_penalty=1.2, 
            eos_token_id=self.special_tokens["end_of_speech"],
            )
        
        self._update_existing_embeds(output_tokens)

        print("successffully updated existing embeds")

        print(output_tokens.shape)

        output = self.parent.parse_output_tokens(output_tokens)
        return output
        



class OrpheusUtility():
    def __init__(self,
                 text_model_name="amuvarma/3b-zuckreg-convo",
                 multimodal_model_name="amuvarma/zuck-3bregconvo-automodelcompat"
                 ):
        self.special_tokens = {
            "start_of_text": 128000,
            "end_of_text": 128009,
            "start_of_speech": 128257,
            "end_of_speech": 128258,
            "start_of_human": 128259,
            "end_of_human": 128260,
            "start_of_ai": 128261,
            "end_of_ai": 128262,
            "pad_token": 128263
        }
        self._is_model_initialised = False
        self._is_model_downloaded = False
        self.audio_encoder = whisper.load_model("small")

        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        snac_model = snac_model.to("cuda")
        self.snac_model = snac_model

        pass

    def _verify_initialisation(self):
        if not self._is_model_initialised:
            raise ValueError("Please ensure you have registered the model with the OrpheusUtility class using orpheus.register_auto_model(model)")
        return self._is_model_initialised

    def _download_from_hub(self, model_name):
        snapshot_download(
            repo_id=model_name,
            allow_patterns=[
                "config.json",
                "*.safetensors",
                "model.safetensors.index.json",
            ],
            ignore_patterns=[
                "optimizer.pt",
                "pytorch_model.bin",
                "training_args.bin",
                "scheduler.pt",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "tokenizer.*"
            ]
        )

    def initialise(self, text_model_name="amuvarma/3b-zuckreg-convo", multimodal_model_name="amuvarma/zuck-3bregconvo-automodelcompat"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_text = executor.submit(
                self._download_from_hub, text_model_name)
            future_multimodal = executor.submit(
                self._download_from_hub, multimodal_model_name)
            future_text.result()
            future_multimodal.result()
        self._is_model_downloaded = True

        AutoConfig.register("orpheus", OrpheusConfig)
        AutoModel.register(OrpheusConfig, OrpheusForConditionalGeneration)


        print("Downloads complete!")

    def register_auto_model(self, model, tokenizer):

        assert model is not None, "You must provide a model"
        assert tokenizer is not None, "You must provide a tokenizer"

        self.model = model
        self.tokenizer = tokenizer
        self._is_model_initialised = True


    def _get_input_from_text(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        input_ids = torch.cat([
            torch.tensor([[self.special_tokens["start_of_human"]]]),
            input_ids, 
            torch.tensor([[self.special_tokens["end_of_text"], 
            self.special_tokens["end_of_human"]]])], 
        dim=1)
        input_ids = input_ids.to("cuda")
        return {"input_ids": input_ids}
    

    def _process_audio_tensor(self, audio, sample_rate=16000):
        audio = audio.to(torch.float32)
        duration_ms = (len(audio) / sample_rate) * 1000
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return mel, int(duration_ms / 20) + 1
    
    def _get_audio_features(self, speech):
        audio_input = speech.squeeze(0)
        mel, length = self._process_audio_tensor(audio_input)
        mel = mel.to("cuda")
        mel = mel.unsqueeze(0)
        audio_feature = self.audio_encoder.embed_audio(mel)[0][:length]
        audio_feature = audio_feature.unsqueeze(0)
        return audio_feature
    
    def _get_input_from_speech(self, speech):
        self._verify_initialisation()
        audio_features = self._get_audio_features(speech)
        audio_features = audio_features.to(dtype=torch.bfloat16).to("cuda")
        audio_embeds = self.model.multi_modal_projector(audio_features)
        start_token = torch.tensor([[self.special_tokens["start_of_human"], self.special_tokens["start_of_text"]]], dtype=torch.int64)
        end_tokens = torch.tensor([[self.special_tokens["end_of_text"], self.special_tokens["end_of_human"]]], dtype=torch.int64)
        start_token = start_token.to("cuda")
        end_tokens = end_tokens.to("cuda")
        start_embeds = self.model.get_input_embeddings()(start_token)
        end_embeds = self.model.get_input_embeddings()(end_tokens)
        start_embeds = start_embeds.to(dtype=torch.bfloat16)
        end_embeds = end_embeds.to(dtype=torch.bfloat16)
        all_embeds = torch.cat([start_embeds, audio_embeds, end_embeds], dim=1)
        return {"inputs_embeds": all_embeds}
    
    def get_inputs(self, text=None, speech=None):
        if text is None and speech is None:
            raise ValueError("Either text or speech must be provided")
        if text is not None and speech is not None:
            raise ValueError("Only one of text or speech must be provided")
        

        if text is not None:
            return self._get_input_from_text(text)
        else:
            return self._get_input_from_speech(speech)
        
    
    def _redistribute_codes(self, code_list):
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))

        codes = [torch.tensor(layer_1).unsqueeze(0).to("cuda"),
                torch.tensor(layer_2).unsqueeze(0).to("cuda"),
                torch.tensor(layer_3).unsqueeze(0).to("cuda")]
        audio_hat = self.snac_model.decode(codes)
        return audio_hat
    
    def _get_waveform_from_tokens (self, output_tokens):

        token_to_find = self.special_tokens["start_of_speech"]
        token_to_remove = self.special_tokens["pad_token"]

        if token_to_find not in output_tokens:
            return None
        
        token_indices = (output_tokens == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[-1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = output_tokens[:, last_occurrence_idx+1:]
        else:
            cropped_tensor = output_tokens

        mask = cropped_tensor != token_to_remove
        cropped_tensor = cropped_tensor[mask].view(cropped_tensor.size(0), -1)

        processed_tensor = cropped_tensor - 128266
        original_shape = processed_tensor.shape
        new_dim_1 = (original_shape[1] // 7) * 7
        processed_tensor = processed_tensor[:, :new_dim_1]
        code_list = processed_tensor[0].tolist()
        samples = self._redistribute_codes(code_list)

        waveform = samples.detach().squeeze().to("cpu").numpy()
        return waveform
        
    def _get_text_from_tokens(self, output_tokens):
        
        token_to_find = self.special_tokens["start_of_ai"]
        end_token = self.special_tokens["end_of_text"]

        print("getting text")

        if token_to_find not in output_tokens:
            return None

        token_indices = (output_tokens == token_to_find).nonzero(as_tuple=True)
        
        print("token indices", token_indices)

        if len(token_indices[1]) > 0:

            start_idx = token_indices[1][-1].item() + 1
            end_indices = (output_tokens[:, start_idx:] == end_token).nonzero(as_tuple=True)
            
            print("end indices", end_indices)
            if len(end_indices[1]) > 0:
                end_idx = start_idx + end_indices[1][0].item()
                text_tokens = output_tokens[:, start_idx:end_idx]
                decoded_text = self.tokenizer.decode(text_tokens[0])
                print("decoded text", decoded_text)
                return decoded_text
                
        return None


    def parse_output_tokens (self, output_tokens):
        waveform = self._get_waveform_from_tokens(output_tokens)
        print(waveform)
        text = self._get_text_from_tokens(output_tokens)
        response_dict = {"waveform": waveform, "text": text}
        if waveform is None:
            response_dict["waveform"] = "No tokens generated to output speech, please increase number of tokens generated"
        if text is None:
            response_dict["text"] = "No tokens generated to output text. There may be an error, or the number of tokens this model is set to generate is too low."
        return response_dict


    def initialise_conversation_model(self):
        return OrpheusConversation(self)
    
    def get_dummy_speech_link(self, speech_path="https://firebasestorage.googleapis.com/v0/b/speechsdk.appspot.com/o/speech.wav?alt=media&token=79d7119b-ab42-4652-b045-ad9a0578ec97"):
        return speech_path
