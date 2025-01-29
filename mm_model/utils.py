from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModel
import torch


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

    def fast_download_from_hub(self, text_model_name="amuvarma/3b-zuckreg-convo", multimodal_model_name="amuvarma/zuck-3bregconvo-automodelcompat"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_text = executor.submit(
                self._download_from_hub, text_model_name)
            future_multimodal = executor.submit(
                self._download_from_hub, multimodal_model_name)
            future_text.result()
            future_multimodal.result()

        print("Downloads complete!")

    def _initialise_model(self, multimodal_model_id):
        if not self.is_model_downloaded:
            self.fast_download_from_hub()
            print("Downloading model from hub...")

        if not self.is_model_initialised:
            self.model = AutoModel.from_pretrained(
                multimodal_model_id, config=self.config, new_vocab_size=False).to(dtype=torch.bfloat16).to("cuda")
            self.is_model_initialised = True

    def get_inputs_from_text(self, text):
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        print("inputs", inputs)
        input_ids = torch.cat([torch.tensor([self.special_tokens["start_of_human"]]),
                              input_ids, torch.tensor([self.special_tokens["end_of_text"], self.special_tokens["end_of_human"]])], dim=1)
        return {"input_ids": input_ids}

    def initialise_conversation_model(self):
        return OrpheusConversation()
