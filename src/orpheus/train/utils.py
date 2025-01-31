from huggingface_hub import HfApi, snapshot_download
from datasets import load_dataset
from .stage_1 import Stage_1_Trainer
from .stage_2 import Stage_2_Trainer
from .stage_3 import Stage_3_Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer

class OrpheusTrainer():
    def _load_dataset(self, dataset_name):
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",   
            revision="main",        
            max_workers=64         
        )
        return load_dataset(dataset_name, split="train")

    def _load_model(self, model_name):
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

        return AutoModelForCausalLM.from_pretrained(model_name)
    
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
                    transcription_dataset = "amuvarma/mls-eng-10k-500k-projection_prep",
                    qa_dataset = None
                ):

        self.use_wandb = use_wandb


        
        if model_name is not None:
            self.model = self._load_model(model_name)
        
        if tokenizer_name is not None:
            self.tokenizer = self._load_tokenizer(model_name)
        
        if base_model_name is not None:
            self.base_model = self._load_model(base_model_name)

        if dataset_name is not None:
            self.dataset = self._load_dataset(dataset_name)
        
        if text_dataset_name is not None:
            self.text_dataset = self._load_dataset(text_dataset_name)
        
        if speech_dataset_name is not None:
            self.speech_dataset = self._load_dataset(speech_dataset_name)

        if transcription_dataset is not None:
            self.transcription_dataset = self._load_dataset(transcription_dataset)

        if qa_dataset is not None:
            self.qa_dataset = self._load_dataset(qa_dataset)

        

        self.pad_token = pad_token
        
        assert stage in ["stage_1", "stage_2", "stage_3", "stage_4"], "Please pass valid stage."

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
            assert base_model_name is not None, "Please pass the name of the model you trained in stage 1 or stage 2 if you chose to do stage 2."

            self._training_class = Stage_3_Trainer(
                model = self.model,
                transcription_dataset = self.transcription_dataset,
                qa_dataset = self.qa_dataset,
                tokenizer = self.tokenizer,
                pad_token = self.pad_token
            )
        
        if stage == "stage_4":
            assert dataset_name is not None, "Please pass the name of the processed dataset."
            assert model_name is not None, "Please pass model_name you trained in stage 3."
            assert base_model_name is not None, "Please pass the name of the model you trained in stage 1 or stage 2 "

        self._training_stage = stage

    

    