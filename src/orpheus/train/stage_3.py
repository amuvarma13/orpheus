import torch
import numpy as np
from .components import InterleavedFSDPTrainer, BatchedAlternatingDataset
from transformers import TrainingArguments
from snac import SNAC
import torchaudio.transforms as T
from collections import defaultdict
from datasets import load_dataset, Dataset

class Stage_3_Trainer():
    def __init__(
            self, 
            model,  
            dataset = None, 
            tokenizer = None,
            save_folder = "checkpoints",
            pad_token = None, 
            max_length = 9600,

        ):
        self.num_threads = 1
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length

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

        #get cuda device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)
        self.dataset = dataset

        self._process_dataset(self.dataset)
        
        self.dataset = BatchedAlternatingDataset(self.processed_text_dataset, self.processed_speech_dataset, batch_total=self.batch_size*self.num_gpus)
        pass





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
            