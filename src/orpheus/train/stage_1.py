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

import multiprocessing


class Stage_1_Trainer():
    def __init__(
            self, 
            model,  
            text_dataset=None, 
            speech_dataset = None, 
            tokenizer = None,
            save_folder = "checkpoints",
            pad_token = None

        ):
        self.text_dataset = text_dataset
        self.num_threads = 1
        self.tokenizer = tokenizer

        self.speech_dataset = speech_dataset
        self.model = model

        self.pad_token = pad_token
        
        # some default values that can be overridden in the .train() method
        self.batch_size = 1
        self.epochs = 1
        self.save_steps = 2000
        self.learning_rate = 5.0e-6

        self.num_gpus = torch.cuda.device_count()
        self.tokeniser_length = 128256
        self.start_of_text = 128000
        self.end_of_text = 128009

        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2

        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4

        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7

        self.start_of_system = self.tokeniser_length + 8
        self.end_of_system = self.tokeniser_length + 9

        self.audio_tokens_start = self.tokeniser_length + 10

        if pad_token is None:
            self.pad_token = 128263


        self.save_folder = save_folder

        self.processed_text_dataset = self._process_text_dataset(self.text_dataset)
        self.dataset = BatchedAlternatingDataset(self.processed_text_dataset, speech_dataset, batch_total=self.batch_size*self.num_gpus)


        pass

    def _create_question_tokens(self, example):
        text_tokens = self.tokenizer.encode(example['question'], add_special_tokens=True)
        text_tokens.append(self.end_of_text)  # Append token 1 to the end
        return {'question_text': text_tokens}
    
    def _create_answers_tokens(self, example):
        text_tokens = self.tokenizer.encode(example['answer'], add_special_tokens=True)
        text_tokens.append(self.end_of_text)  # Append token 1 to the end
        return {'answer_text': text_tokens}
    
    def _create_input_ids(self, example):
        input_ids = (

            [self.start_of_human] +
            example['question_text'] +
            [self.end_of_human] +
            [self.start_of_ai] +
            example['answer_text']
        )

        example['input_ids'] = input_ids
        example["attention_mask"] = [1] * len(input_ids)
        example["labels"] = input_ids
        return example

    def _process_text_dataset(self, text_dataset):

        text_dataset = text_dataset.map(
                self._create_question_tokens,
                num_proc=self.num_threads,
                desc="Preprocessing your text dataset, Step 1 of 3",
            )
    
        text_dataset = text_dataset.map(
            self._create_answers_tokens,
            num_proc=self._num_threads,
            desc="Preprocessing your text dataset, Step 2 of 3",
        )

        text_dataset = text_dataset.map(
            self._create_input_ids,
            num_proc=self._num_threads,
            desc="Preprocessing your text dataset, Step 3 of 3",
        )

        columns_to_keep = ["input_ids", "attention_mask", "labels"]
        all_columns = text_dataset.column_names
        columns_to_remove = [col for col in all_columns if col not in columns_to_keep]

        return text_dataset.remove_columns(columns_to_remove)


    
    def _create_training_args (self, **kwargs):
        

        assert self.num_gpus > 1, "At least 2 GPUs should be available for training, to allow FSDP."

        self.training_args = TrainingArguments(
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size, 
            logging_steps=1,
            bf16=True,
            output_dir=f"./{self.save_folder}",
            fsdp="auto_wrap",
            save_steps=self.save_steps,
            remove_unused_columns=True, 
            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine", 
            **kwargs
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
            **kwargs
        ):
        self._create_training_args(**kwargs)
        trainer = InterleavedFSDPTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            compute_metrics=self._compute_metrics,
            data_collator=self._data_collator,
        )
        return trainer
            