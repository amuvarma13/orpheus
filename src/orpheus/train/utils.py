from huggingface_hub import HfApi, snapshot_download
from datasets import load_dataset
from .stage_1 import Stage_1_Trainer
from .stage_2 import Stage_2_Trainer
from .stage_3 import Stage_3_Trainer
from .stage_4 import Stage_4_Trainer
from .stage_5 import Stage_5_Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..model import OrpheusForConditionalGeneration
from ..config import OrpheusConfig
from ..utils import OrpheusUtility
import torch
from transformers import AutoModel

class OrpheusTrainer():
    def _load_dataset(self, dataset_name):
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",   
            revision="main",        
            max_workers=64         
        )
        return load_dataset(dataset_name, split="train")
    
    def _download_model(self, model_name):
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

    def _load_model(self, model_name):
        self._download_model(model_name)
        return AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")
    
    def _load_orpheus_model(self, model_name):
        self._download_model(model_name)
        config = OrpheusConfig(
            text_model_id=model_name,
            audio_token_index=156939,
            vocab_size=156939,
        )

        model = OrpheusForConditionalGeneration(config)

        return model

    def _load_orpheus_model_from_orpheus(self, model_name):
        orpheus_utility = OrpheusUtility()
        orpheus_utility.initialise()
        self._download_model(model_name)
        model = AutoModel.from_pretrained(model_name).to("cuda").to(torch.bfloat16)
        return model


    
    def _load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def create_trainer(self, **kwargs):
        return self._training_class.create_trainer(**kwargs)

    def __init__ (self, 
                    stage="stage_1", 
                    text_dataset_name="amuvarma/stage_1_training_example",
                    speech_dataset_name="amuvarma/stage_1_training_example",
                    dataset_name = None,
                    use_wandb = True,
                    model_name = "amuvarma/3b-10m-pretrain-full", 
                    base_model_name = None,
                    pad_token = None, 
                    tokenizer_name = "amuvarma/3b-10m-pretrain-full", 
                    batch_size = None,
                    dataset = None
                ):

        assert dataset is None or dataset_name is None, "Please pass either dataset or dataset_name, not both."

        self.dataset = None
        if dataset is not None:
            self.dataset = dataset

        self.use_wandb = use_wandb
        
        if model_name is not None:
            if stage == "stage_1" or stage == "stage_2":
                self.model = self._load_model(model_name)
                self.model = self.model.to(torch.bfloat16)

            elif stage == "stage_3":
                self.model = self._load_orpheus_model(model_name)
            elif stage == "stage_4" or stage == "stage_5":
                self.model = self._load_orpheus_model_from_orpheus(model_name)

        elif stage == "stage_1":
            model_name = "amuvarma/3b-10m-pretrain-full"
            self.model = self._load_model(model_name)
            
        
        if tokenizer_name is not None:
            self.tokenizer = self._load_tokenizer(model_name)
        
        if base_model_name is not None:
            if stage == "stage_3" or stage == "stage_4":
                self.base_model = self._load_model(base_model_name)

        if dataset_name is not None:
            if stage == "stage_2" or stage == "stage_4":
                self.dataset = self._load_dataset(dataset_name)
                self.dataset = self.dataset.shuffle(seed=42)
                # self.dataset = self.dataset.select(range(100))

        
        if text_dataset_name is not None:
            if stage == "stage_1":
                self.text_dataset = self._load_dataset(text_dataset_name)
        
        if speech_dataset_name is not None:
            if stage == "stage_1":
                self.speech_dataset = self._load_dataset(speech_dataset_name)


        

        self.pad_token = pad_token
        
        assert stage in ["stage_1", "stage_2", "stage_3", "stage_4", "stage_5"], "Please pass valid stage."

        if stage == "stage_1":
            assert text_dataset_name is not None, "Please pass text_dataset_name."
            assert speech_dataset_name is not None, "Please pass speech_dataset_name."

            self._training_class = Stage_1_Trainer(
                model = self.model,  
                text_dataset=self.text_dataset, 
                speech_dataset = self.speech_dataset, 
                pad_token = self.pad_token, 
                tokenizer=self.tokenizer
            )

        if stage == "stage_2":
            assert dataset_name is not None, "Please pass dataset_name."
            assert model_name is not None, "Please pass the name of the model you trained in stage 1."

            self._training_class = Stage_2_Trainer(
                model = self.model,
                dataset = self.dataset,
                tokenizer = self.tokenizer,
                pad_token = self.pad_token
            )


        if stage == "stage_3":
            assert model_name is not None, "Please pass model_name."

            if self.dataset is None:
                self.dataset = self._load_dataset("amuvarma/mls-eng-10k-500k-projection_prep")
            

            self._training_class = Stage_3_Trainer(
                model = self.model,
                dataset = self.dataset,
                tokenizer = self.tokenizer,
                pad_token = self.pad_token, 
                batch_size = batch_size
            )
        
        if stage == "stage_4":
            assert model_name is not None, "Please pass model_name you trained in stage 3."

            if self.dataset is None:
                self.dataset = self._load_dataset("gpt-omni/VoiceAssistant-400K")
            

            self._training_class = Stage_4_Trainer(
                model = self.model,
                dataset = self.dataset,
                tokenizer = self.tokenizer,
                pad_token = self.pad_token,
                batch_size = batch_size
            )

        if stage == "stage_5":
            assert model_name is not None, "Please pass model_name you trained in stage 4."

            self._training_class = Stage_5_Trainer(
                model = self.model,
                dataset = self.dataset,
                tokenizer = self.tokenizer,
                pad_token = self.pad_token,
                batch_size = batch_size
            )

        self._training_stage = stage

    

    