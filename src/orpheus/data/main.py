from scipy.io.wavfile import write
import msinference 
import random
import nltk
import librosa
from datasets import load_dataset, Audio

nltk.download('punkt_tab')
text = 'Hello world!'
voices_strings = ["f-us-1.wav", "f-us-2.wav", "f-us-3.wav", "f-us-4.wav", "m-us-1.wav", "m-us-2.wav", "m-us-3.wav", "m-us-4.wav"]
voices = [msinference.compute_style("voices/"+voice) for voice in voices_strings]
dsn = "amuvarma/snacced-flat-zuck-convo"
push_name = "amuvarma/snacced-flat-zuck-convo-sttsed"

ds = load_dataset(dsn, split='train')
ds = ds.shuffle(seed=42)
# ds = ds.select(range(100))
def add_audio(example):
    try:
        
        text = example['question']
        voice = random.choice(voices)
        wav = msinference.inference(
            text, 
            voice, 
            alpha=0.3, 
            beta=0.7, 
            diffusion_steps=7, 
            embedding_scale=1
        )
        
        wav_16k = librosa.resample(wav, orig_sr=24000, target_sr=16000)
        
        print(wav_16k.shape)
        
        return {
            'answer_audio': {
                'array': wav_16k,
                'sampling_rate': 16000
            }
        }
    except Exception as e:
        print(f"Failed to process example: {e}")
        return {
            'answer_audio': None  # Or you could return a default value
        }

ds = ds.map(add_audio, batched=False)
ds = ds.cast_column("answer_audio", Audio(sampling_rate=16000))


ds.push_to_hub(push_name)
