import random
import librosa
from functools import partial
from datasets import load_dataset
from huggingface_hub import snapshot_download
import torchaudio.transforms as T

class OrpheusDataProcessor():
    def __init__(self):
        self.voices = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
        self.resampler = T.Resample(orig_freq=24000, new_freq=16000)
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
    
    
    def _mark_missing_question_audio(self, example):
    # Return None for examples missing the key
        if "question_audio" not in example or example["question_audio"] is None:
            return None
        return example



    
    def adapt_stage_1_to_stage_5_dataset(self, dataset):
        
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code='b', device="cuda") 

        add_audio_fn = partial(
            self._add_audio,
            column_name='question',
            audio_column_name='question_audio',
            target_sr=16000
        )

        dataset = dataset.map(add_audio_fn, batched=False)
        print("adapting dataset", dataset)
        dataset = dataset.map(self._mark_missing_question_audio)
        print("adapted dataset", dataset)
        dataset = dataset.filter(lambda example: example is not None)
        return dataset

    def _add_audio(self, example, column_name, audio_column_name, target_sr=16000):
        try:
            text = example[column_name]

            example[audio_column_name] = None
            
            voice = random.choice(self.voices)

            generator = self.pipeline(
                text, voice=voice, # <= change voice here
                speed=1, split_pattern=r'\n+'
            )

            (gs, ps, audio) = next(generator)

            resampled_audio = self.resampler(audio)

            example[audio_column_name] = {
                'array': resampled_audio.to('cpu').numpy(),
                'sampling_rate': 16000
            }
            return example
        
        except Exception as e:
            print(f"Failed to process example: {e}")
            return {
                'answer_audio': None  # Or you could return a default value
            }
