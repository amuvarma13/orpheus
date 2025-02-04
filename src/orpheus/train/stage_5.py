import torch
import numpy as np
from .components import InterleavedFSDPTrainer, BatchedAlternatingDataset
from transformers import TrainingArguments
import torchaudio.transforms as T
from collections import defaultdict
from datasets import load_dataset, Dataset
import whisper
from transformers import Trainer
from snac import SNAC



class AudioChatDataCollator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        cuda_device = torch.cuda.current_device()
        print("device of whisper", cuda_device)
        whisper_model = whisper.load_model("small", device=f'cuda:{cuda_device}')
        self.whisper_model = whisper_model
        self.cuda_device = cuda_device
        self.model = model
        self.index = 0


        pass 

    def _process_audio_tensor(self, audio, sample_rate=16000):
        audio = audio.to(torch.float32)
        duration_ms = (len(audio) / sample_rate) * 1000
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return mel, int(duration_ms / 20) + 1

    def _inference_collator(self, audio_input, user_res, ass_res, snac_tokens=[]):
        user_input_ids = self.tokenizer(user_res, return_tensors="pt").input_ids
        assistant_input_ids = self.tokenizer(ass_res, return_tensors="pt").input_ids

        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)
        final_tokens = torch.tensor([[128009, 128257 ]], dtype=torch.int64)
        post_assistant_tokens = torch.tensor([[128258, 128262]])

        # crop to first 5000 tokens of snac list
        
        self.index = self.index + 1
        print(self.index)
        if len(snac_tokens) > 4200:
            snac_tokens = snac_tokens[:4200]

      
        # if index is even
        if self.index % 2 ==0:
            snac_tokens = []


        user_tokens = torch.cat(
            [start_token, user_input_ids, end_tokens], dim=1)
        snac_tokens_tensor = torch.tensor([snac_tokens], dtype=torch.int64)

        if len(snac_tokens) > 0:
            labels = torch.cat([start_token, user_input_ids, end_tokens,
                            assistant_input_ids, final_tokens, snac_tokens_tensor, post_assistant_tokens], dim=1)
            
        else:
            labels = torch.cat([start_token, user_input_ids, end_tokens,
                            assistant_input_ids, final_tokens], dim=1)

        true_labels = torch.full_like(labels, -100)
        true_labels[:, user_tokens.shape[1]:] = labels[:, user_tokens.shape[1]:]

        attention_mask = torch.ones_like(labels)

        audio_input = audio_input.squeeze(0)
        mel, length = self._process_audio_tensor(audio_input)
        
        mel = mel.to(f'cuda:{self.cuda_device}')
        mel = mel.unsqueeze(0)
        audio_feature = self.whisper_model.embed_audio(mel)[0][:length]
        audio_feature = audio_feature.unsqueeze(0)

        return {
            "audio_values": audio_feature.to(self.model.device).to(self.model.dtype),
            "input_ids": labels.to(self.model.device),
            "labels": true_labels.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device)
        }

    def __call__(self, features):
        audio = torch.tensor([features[0]["question_audio"]["array"]])
        assistant_response = features[0]["answer"]
        user_response = features[0]["question"]
        snac_tokens = features[0]["codes_list"]

        if "<|audio|>" in user_response:
            user_response = features[0]["answer"]
        else:
            user_response = "<|audio|>"

        batch = self._inference_collator(
            audio, user_response, assistant_response, snac_tokens)

        return {
            "audio_values": batch["audio_values"].cpu(),
            "input_ids": batch["input_ids"].cpu(),
            "labels": batch["labels"].cpu(),
            "attention_mask": batch["attention_mask"].cpu()
        }



class Stage_5_Trainer():
    def __init__(
            self,
            model,
            dataset=None,
            tokenizer=None,
            save_folder="checkpoints",
            pad_token=None,
            max_length=9600,
            batch_size=None,
            processed_dataset=False

    ):
        self.num_threads = 4
        self.tokenizer = tokenizer
        self.model = self._prepare_model(model)
        self.max_length = max_length

        self.pad_token = pad_token

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = 1

        self.num_gpus = torch.cuda.device_count()

        self.gradient_accumulation_steps = 1

        # some default values that can be overridden in the .train() method
        self.epochs = 1
        self.save_steps = 2000
        self.learning_rate = 5.0e-6

        self.tokeniser_length = 128256
        self.start_of_text = 128000
        self.end_of_text = 128009

        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4

        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7

        self.start_of_system = self.tokeniser_length + 8
        self.end_of_system = self.tokeniser_length + 9

        self.audio_tokens_start = self.tokeniser_length + 10
        self.sr = 16000

        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)


        if pad_token is None:
            self.pad_token = 128263

        self.save_folder = save_folder

        # get cuda device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = dataset
        if processed_dataset:
            self.processed_dataset = dataset
        else:
            self.processed_dataset = self._add_codes_to_dataset(self.dataset)

        pass

    def _prepare_model(self, model):
        model = model.to(torch.bfloat16)
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "multi_modal_projector" in name:
                param.requires_grad = True
        
        return model
    
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
    
    def _add_codes_to_dataset(self, dataset):
        self.sr = dataset[0]["answer_audio"]["sampling_rate"]
        dataset = dataset.map(self._add_codes)
        return dataset
    

    def _create_training_args(self, **kwargs):

        self.training_args = TrainingArguments(
            output_dir=self.save_folder,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=2e-3,
            logging_steps=1,
            fsdp = True,
            evaluation_strategy="no",
            push_to_hub=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            bf16=True,
            save_steps=15000,
            **kwargs
        )


    def create_trainer(
        self,
        **kwargs
    ):
        print("about to load whisper", self.model.device)
        #self.device is get torch cuda device

        # self.whisper_model = whisper_model.to(self.device)

        self._create_training_args(**kwargs)
        print("processed ds", self.processed_dataset)
        trainer = InterleavedFSDPTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.processed_dataset,
            data_collator=AudioChatDataCollator(self.tokenizer,self.model),
        )
        return trainer
