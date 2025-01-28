from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor


class OrpheusConversation():
    def __init__(self):
        self.message_embeds = []


class OrpheusUtility():
    def __init__(self):
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

    def fast_download_from_hub(self, text_model_name="amuvarma/3b-zuckreg-convo", multimodal_model_name="amuvarma/zuck-3bregconvo-automodelcompat"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_text = executor.submit(self._download_from_hub, text_model_name)
            future_multimodal = executor.submit(self._download_from_hub, multimodal_model_name)
            future_text.result()
            future_multimodal.result()
        
        print("Downloads complete!")


    def initialise_conversation_model(self):
        return OrpheusConversation()