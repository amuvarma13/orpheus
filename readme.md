# Inference

### Setup Environment

#### 1. Installation
Clone this repository.
```bash
git clone https://github.com/amuvarma13/orpheus.git
pip install snac
```


#### 2. Additional setup
If you are running this model on Google Colab you can skip this step. Cuda version should be 12.X otherwise you will have conflicts with the version of `flash_attn`. These can be resolved by reverting to an earlier version of `flash_attn`. 
```bash
pip install torch==2.5.1 torchaudio transformers==4.47.0 flash_attn librosa soundfile
```

Colab will have appropriate versions of these packages installed.


#### 3. Import relevant Orpheus modules

Due to how colab processes modules if you are on Colab import the  correct version.
```python
from orpheus.mm_model_from_colab.model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
)
from orpheus.mm_model_from_colab.utils import fast_download_from_hub
from orpheus.mm_model_from_colab.utils import parse_output_tokens
```
If you are running this on a normal VM import the default version
```python
from mm_model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
    fast_download_from_hub, 
    format_text_input,
    format_speech_input,
    format_conversation,
    parse_output_tokens
)
```


#### 4. Register the model type with transformers

Now we register the model so that we can use it with AutoModel and AutoTokenizer.

```python
from transformers import AutoModel, AutoTokenizer, AutoConfig
AutoConfig.register("orpheus", OrpheusConfig)
AutoModel.register(OrpheusConfig, OrpheusForConditionalGeneration)
```

5. Instantiate Model
```python
fast_download_from_hub() 
model_name = "amuvarma/zuck-3bregconvo-automodelcompat"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Setup Environment

The model can accept both text and speech inputs and outputs both text and speech outputs. You can use this model much like any LLM found on huggingface transformers.

This section will show you how to run inference on text inputs, speech inputs, or multiturn conversations with combined inputs. We use a standard format for chats with ```start_of_human```, ```end_of_human```, ```start_of_ai```, and ```end_of_ai``` tokens to guide the model to understand whose turn it is.

#### Text input

First we can pass our text based question as follows to generate our output tokens. We use the utility function provided which adds a couple of extra tokens to indicate the structure to the model.

```python
prompt = "What is an example of a healthy breakfast?"
inputs = tokenizer.encode(prompt, return_tensors="pt")
output_tokens = model.generate(
    **inputs, 
    max_new_tokens=2000, 
    repetition_penalty=1.1, 
    temperature=0.7
    )
```

Next we can parse our output tokens to get both text and speech responses using the helper function provided which we imported earlier.

```python
output_text, output_speech = parse_output_tokens(output_tokens)
print(f"Response is {output_text}")
print(f"Shape of speech, {output_speech.shape}")
```

The output speech is a numpy array of samples, which we can display using iPython if we are in a notebook environment, or save to a file.

``` python
# display using iPython
from IPython.display import Audio, display
display(Audio(output_speech, rate=16000))

#save to file
import soundfile as sf
sf.write("output.wav", output_speech, 16000)
```
