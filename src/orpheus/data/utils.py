import random
import librosa
from .msinference import compute_style, inference
from functools import partial
from datasets import load_dataset
from huggingface_hub import snapshot_download

class OrpheusDataProcessor():
    def __init__(self):
        self.voices = self._compute_voices()
        pass

    def fast_load_dataset(self, dataset_name, split="train"):
        dataset = self._load_dataset(dataset_name, split=split)
        return dataset

    def _load_dataset(self, dataset_name, split="train"):
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",   
            revision="main",        
            max_workers=64         
        )
        return load_dataset(dataset_name, split=split)
    
    def adapt_stage_1_to_stage_5_dataset(self, dataset):
        add_audio_fn = partial(
            self._add_audio,
            column_name='question',
            audio_column_name='question_audio',
            target_sr=16000
        )

        dataset = dataset.map(add_audio_fn, batched=False)
        return dataset



    def _compute_voices(self):
        voices_strings = ["f-us-1.wav", "f-us-2.wav", "f-us-3.wav", "f-us-4.wav", "m-us-1.wav", "m-us-2.wav", "m-us-3.wav", "m-us-4.wav"]
        voices = [compute_style("voices/"+voice) for voice in voices_strings]
        return voices

    def _add_audio(self, example, column_name, audio_column_name, target_sr=16000):
        try:
            text = example[column_name]
            voice = random.choice(self.voices)
            wav = inference(
                text, 
                voice, 
                alpha=0.3, 
                beta=0.7, 
                diffusion_steps=7, 
                embedding_scale=1
            )
            
            wav_16k = librosa.resample(wav, orig_sr=24000, target_sr=target_sr)

            example[audio_column_name] = {
                'audio': wav_16k,
                'sampling_rate': 16000
            }
            return example
        
        except Exception as e:
            print(f"Failed to process example: {e}")
            return {
                'answer_audio': None  # Or you could return a default value
            }
