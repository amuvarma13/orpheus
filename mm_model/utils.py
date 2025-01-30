from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModel
import torch
import whisper
from transformers import AutoModel, AutoConfig

from .model import OrpheusConfig, OrpheusForConditionalGeneration




class OrpheusConversation():
    def __init__(self):
        self.message_embeds = []


class OrpheusUtility():
    def __init__(self,
                 text_model_name="amuvarma/3b-zuckreg-convo",
                 multimodal_model_name="amuvarma/zuck-3bregconvo-automodelcompat"
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(multimodal_model_name)
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
        self.is_model_initialised = False
        self.is_model_downloaded = False
        self.audio_encoder = whisper.load_model("small")
        pass

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
        self.is_model_downloaded = True

    def initialise(self, text_model_name="amuvarma/3b-zuckreg-convo", multimodal_model_name="amuvarma/zuck-3bregconvo-automodelcompat"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_text = executor.submit(
                self._download_from_hub, text_model_name)
            future_multimodal = executor.submit(
                self._download_from_hub, multimodal_model_name)
            future_text.result()
            future_multimodal.result()

        AutoConfig.register("orpheus", OrpheusConfig)
        AutoModel.register(OrpheusConfig, OrpheusForConditionalGeneration)


        print("Downloads complete!")


    def _initialise_model(self, multimodal_model_id):
        if not self.is_model_downloaded:
            self.fast_download_from_hub()
            print("Downloading model from hub...")

        if not self.is_model_initialised:
            self.model = AutoModel.from_pretrained(
                multimodal_model_id, config=self.config, new_vocab_size=False).to(dtype=torch.bfloat16).to("cuda")
            self.is_model_initialised = True


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
        audio = self.audio_encoder.pad_or_trim(audio)
        mel = self.audio_encoder.log_mel_spectrogram(audio)
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

        audio_features = speech.to(dtype=torch.bfloat16).to("cuda")
        audio_embeds = self.model.multi_modal_projector(audio_features)
        start_token = torch.tensor([[self.special_tokens["start_of_human"]]], dtype=torch.int64)
        end_tokens = torch.tensor([[self.special_tokens["end_of_text"], self.special_tokens["end_of_human"], self.special_tokens["start_of_ai"]]], dtype=torch.int64)
        final_tokens = torch.tensor([[128262]], dtype=torch.int64)
        start_token = start_token.to("cuda")
        end_tokens = end_tokens.to("cuda")
        final_tokens = final_tokens.to("cuda")
        start_embeds = self.model.get_input_embeddings()(start_token)
        end_embeds = self.model.get_input_embeddings()(end_tokens)
        final_embeds = self.model.get_input_embeddings()(final_tokens)
        start_embeds = start_embeds.to(dtype=torch.bfloat16)
        end_embeds = end_embeds.to(dtype=torch.bfloat16)
        final_embeds = final_embeds.to(dtype=torch.bfloat16)
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


    def initialise_conversation_model(self):
        return OrpheusConversation()
