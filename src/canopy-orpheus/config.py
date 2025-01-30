from transformers import PretrainedConfig, CONFIG_MAPPING

class OrpheusConfig(PretrainedConfig):
    
    model_type = "orpheus"

    is_composition = False
    def __init__(
        self,
        text_config=None,
        audio_hidden_size = 768,
        ignore_index=-100,
        audio_token_index=32000,
        vocab_size=32000,
        hidden_size=3072,
        stack_factor=8,
        projector_type="mlp",
        
        
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.audio_token_index = audio_token_index
        self.vocab_size = vocab_size
        self.audio_hidden_size = audio_hidden_size
        self.text_config = text_config

        self.text_model_id = None

        self.hidden_size = hidden_size
        self.stack_factor = stack_factor
        self.projector_type = projector_type

        if isinstance(self.text_config, dict):
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "llama"
            )
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            self.vocab_size = self.text_config.vocab_size
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["llama"]()
        
        super().__init__(**kwargs)
