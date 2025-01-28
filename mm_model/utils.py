from huggingface_hub import snapshot_download

def _download_from_hub(model_name):

    print("provided model name", model_name)
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

def fast_download_from_hub(text_model_name="amuvarma/3b-zuckreg-convo", multimodal_model_name="amuvarma/zuck-3bregconvo-automodelcompat"):
    print(text_model_name, multimodal_model_name)
    _download_from_hub(text_model_name)
    _download_from_hub(multimodal_model_name)

class OrpheusConversation():
    def __init__(self):
        self.message_embeds = []


class OrpheusUtility():
    def __init__(self):
        pass
    fast_download_from_hub = fast_download_from_hub

    def initialise_conversation_model(self):
        return OrpheusConversation()