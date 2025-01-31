import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import os
import yaml
from .components import InterleavedFSDPTrainer, BatchedAlternatingDataset
from transformers import AutoModel, TrainingArguments


class Stage_1_Trainer():
    def __init__(
            self, 
            model,  
            text_dataset=None, 
            speech_dataset = None, 
            save_folder = "checkpoints",
            pad_token = None

        ):
        self.text_dataset = text_dataset
        self.speech_dataset = speech_dataset
        self.model = model

        self.pad_token = pad_token
        
        # some default values that can be overridden in the .train() method
        self.batch_size = 1
        self.epochs = 1
        self.save_steps = 2000
        self.learning_rate = 5.0e-6

        self.num_gpus = torch.cuda.device_count()

        self.dataset = BatchedAlternatingDataset(text_dataset, speech_dataset, batch_total=self.batch_size*self.num_gpus)


        if pad_token is None:
            self.pad_token = 128263


        self.save_folder = save_folder
        pass
    
    def _calculate_default_hyperparameters(self):
        

        assert self.num_gpus > 1, "At least 2 GPUs should be available for training, to allow FSDP."

        self.training_args = TrainingArguments(
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size, 
            logging_steps=1,
            bf16=True,
            output_dir=f"./{self.save_folder}",
            fsdp="auto_wrap",
            report_to=self.report_to, 
            save_steps=self.save_steps,
            remove_unused_columns=True, 
            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine"
        )
    
    def _data_collator(self, features):
        input_ids = [f["input_ids"] for f in features]

        if any("attention_mask" not in f for f in features):
            attention_mask = [[1]*len(ids) for ids in input_ids]
        else:
            attention_mask = [f["attention_mask"] for f in features]

        if any("labels" not in f for f in features):
            labels = input_ids
        else:
            labels = [f["labels"] for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(i, dtype=torch.long) for i in input_ids], batch_first=True, padding_value=self.pad_token)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(m, dtype=torch.long) for m in attention_mask], batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(l, dtype=torch.long) for l in labels], batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        
    def _compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy} 


    
    def create_trainer(
            self,
        ):
        self._calculate_default_hyperparameters()
        trainer = InterleavedFSDPTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            compute_metrics=self._compute_metrics,
            data_collator=self._data_collator,
        )
        return trainer
            