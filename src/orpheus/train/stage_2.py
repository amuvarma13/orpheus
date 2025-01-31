import torch
import numpy as np
from .components import InterleavedFSDPTrainer, BatchedAlternatingDataset
from transformers import TrainingArguments
from snac import SNAC
import torchaudio.transforms as T
from collections import defaultdict
from datasets import load_dataset, Dataset

class Stage_2_Trainer():
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

        self._process_dataset(self.speech_dataset)
        
        self.dataset = BatchedAlternatingDataset(self.processed_text_dataset, self.processed_speech_dataset, batch_total=self.batch_size*self.num_gpus)


        pass


    def _tokenise_audio(self, waveform):
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)

        resample_transform = T.Resample(orig_freq=self.sr, new_freq=24000)
        waveform = resample_transform(waveform)

        waveform = waveform.unsqueeze(0).to("cuda")
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
    

    def _dataset_to_list_of_lists(self, dataset):
        conv_dict = defaultdict(list)
        
        for row in dataset:
            conv_index = row["conversation_index"]
            conv_dict[conv_index].append({
                "messages_index": row["messages_index"],
                "question": row["question"],
                "answer": row["answer"],
                "codes_list": row["codes_list"],
            })
        
        # Sort each conversation by messages_index, then collect into a list
        result = []
        for conv_index in sorted(conv_dict.keys()):
            messages_sorted = sorted(conv_dict[conv_index], key=lambda x: x["messages_index"])
            result.append(messages_sorted)
        
        return result
    
    def _assemble_input_ids(self, lists):
        all_input_ids = []
        for convo in lists:
            input_ids = []
            for message in convo:
                question = message["question"]
                answer = message["answer"]
                codes_list = message["codes_list"]
                tokenised_question = self.tokenizer.encode(question, add_special_tokens=True)
                tokenised_answer = self.tokenizer.encode(answer, add_special_tokens=True)
                tokenised_question.append(self.end_of_text)
                tokenised_answer.append(self.end_of_text)
                input_ids.extend([self.start_of_human] + tokenised_question + [self.end_of_human] + [self.start_of_ai] + tokenised_answer + [self.start_of_speech] + codes_list + [self.end_of_speech] + [self.end_of_ai])
            all_input_ids.append(input_ids)
        
        return all_input_ids

    def _convert_to_hf_dataset(self, all_input_ids):
        flat_input_ids = [iids for iids in all_input_ids]
        ds = Dataset.from_dict({"input_ids": flat_input_ids})
        return ds
    
    def create_mask_and_labels(self, example):

        if len(example['input_ids']) > self.max_length:
            example['attention_mask'] = [1] * self.max_length
            example['input_ids'] = example['input_ids'][:self.max_length]
        else:
            example['attention_mask'] = [1] * len(example['input_ids'])

        example['labels'] = example['input_ids']
        
        return example


    def _preserve_patches(self, example):
        input_ids = example['input_ids']
        text_labels = [-100] * len(input_ids)
        inside_patch = False

        for i, token in enumerate(input_ids):
            if token == self.start_of_ai:
                inside_patch = True
            if inside_patch:
                text_labels[i] = token
            if token == self.end_of_text:
                inside_patch = False

        example['labels'] = text_labels
        return example
    
    def _create_input_ids (self, dataset):
        lists = self._dataset_to_list_of_lists(dataset)
        all_input_ids = self._assemble_input_ids(lists)
        processed_dataset = self._convert_to_hf_dataset(all_input_ids)
        processed_dataset = processed_dataset.map(self._create_mask_and_labels, num_proc=1)
        processed_dataset_length = len(processed_dataset)
        processed_dataset_text = dataset.select(range(processed_dataset_length//2))
        processed_dataset_speech = dataset.select(range(processed_dataset_length//2, processed_dataset_length))
        processed_dataset_text = processed_dataset_text.map(self._preserve_patches, num_proc=1)

        self.processed_dataset_text = processed_dataset_text
        self.processed_dataset_speech = processed_dataset_speech
        pass


        
    def _process_dataset(self, dataset):
        self.sr = dataset[0]["answer_audio"]["sampling_rate"]
        dataset = dataset.map(self._add_codes, remove_columns=["answer_audio"], desc="Processing speech dataset, Step 1 of 3")
        dataset = dataset.filter(lambda x: x['question'] and x['answer'] and x['codes_list'])
        dataset = dataset.filter(lambda x: len(x['codes_list']) < self.max_length)
        self._create_input_ids(dataset)


    
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
            