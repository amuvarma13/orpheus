from huggingface_hub import snapshot_download




class OrpheusConversation():
    def __init__(self):
        self.message_embeds = []


class OrpheusUtility():
    def __init__(self):
        pass
    def _download_from_hub(self, model_name):
        model_path = snapshot_download(
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
        self._download_from_hub(text_model_name)
        self._download_from_hub(multimodal_model_name)
        print("Downloads complete!")


    def initialise_conversation_model(self):
        return OrpheusConversation()