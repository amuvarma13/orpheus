import torch
import numpy as np
from .components import InterleavedFSDPTrainer, BatchedAlternatingDataset
from transformers import TrainingArguments
from snac import SNAC
import torchaudio.transforms as T


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

        #get cuda device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)

        self.processed_speech_dataset = self._process_speech_dataset(self.speech_dataset)
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
            num_proc=self.num_threads,
            desc="Preprocessing your text dataset, Step 2 of 3",
        )

        text_dataset = text_dataset.map(
            self._create_input_ids,
            num_proc=self.num_threads,
            desc="Preprocessing your text dataset, Step 3 of 3",
        )

        columns_to_keep = ["input_ids", "attention_mask", "labels"]
        all_columns = text_dataset.column_names
        columns_to_remove = [col for col in all_columns if col not in columns_to_keep]

        print(text_dataset[0]["input_ids"])

        return text_dataset.remove_columns(columns_to_remove)
    
    def _tokenise_audio(self, waveform):
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)

        # resample_transform = T.Resample(orig_freq=original_sample_rate, new_freq=16000)
        # waveform = resample_transform(waveform)
        resample_transform = T.Resample(orig_freq=self.sr, new_freq=24000)
        waveform = resample_transform(waveform)

        waveform = waveform.unsqueeze(0).to("cuda")
        #generate the codes from snac
        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item()+128266)
            all_codes.append(codes[1][0][2*i].item()+128266+4096)
            all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))


        return all_codes
    
    def _add_codes(self, example):
        codes_list = None

        try:
            answer_audio = example.get("answer_audio")
            if answer_audio and "array" in answer_audio:
                audio_array = answer_audio["array"]
                codes_list = self._tokenise_audio(audio_array)
        except Exception as e:
            print(f"Skipping row due to error: {e}")

        return {"codes_list": codes_list}

        
    def _process_speech_dataset(self, speech_dataset):
        self.sr = speech_dataset[0]["answer_audio"]["sampling_rate"]
        speech_dataset = speech_dataset.map(self._add_codes, remove_columns=["answer_audio"])
        return speech_dataset
    
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
            