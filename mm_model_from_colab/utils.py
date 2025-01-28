from huggingface_hub import snapshot_download

def _download_from_hub(model_name):
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

def fast_download_from_hub(text_model_name="amuvarma/3b-zuckreg-convo", multimodal_model_name="amuvarma/3b-zuckreg-convo-projsnactune"):
    _download_from_hub(text_model_name)
    _download_from_hub(multimodal_model_name)

    text_model_id= "amuvarma/3b-zuckreg-convo"
    mm_model_id = "amuvarma/3b-zuckreg-convo-projsnactune"

    config = OrpheusConfig(
        text_model_id=model_id,
        audio_token_index=156939,
        vocab_size=156939,
    )
    orpheus = AutoModel.from_pretrained(mm_model_id, config=config, new_vocab_size=False).to(dtype=torch.bfloat16).to("cuda")
    print("Everything is successfully downloaded!")

