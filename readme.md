# Inference

### Setup Environment

#### 1. Installation
Clone this repository.
```bash
git clone https://github.com/amuvarma13/orpheus.git
pip install snac openai-whisper
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
from orpheus.mm_model_from_colab.utils import OrpheusUtility
orpheus = OrpheusUtility()
```
If you are running this on a normal VM import the default version
```python
from mm_model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
    OrpheusUtility
)
orpheus = OrpheusUtility()
```


#### 4. Initialise the model

Now we register the model so that we can use it with AutoModel and AutoTokenizer.

```python
import torch
from transformers import AutoModel, AutoTokenizer

orpheus.initialise()

model_name = "amuvarma/zuck-3bregconvo-automodelcompat"
model = AutoModel.from_pretrained(model_name).to("cuda").to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

orpheus.register_auto_model(model)
```

### Setup Environment

The model can accept both text and speech inputs and outputs both text and speech outputs. You can use this model much like any LLM found on huggingface transformers.

This section will show you how to run inference on text inputs, speech inputs, or multiturn conversations with combined inputs. We use a standard format for chats with ```start_of_human```, ```end_of_human```, ```start_of_ai```, and ```end_of_ai``` tokens to guide the model to understand whose turn it is.

#### Simple Inference (1-turn)

We can pass either text (shown below), speech(shown below), or a combination of text and speech (not shown below) to the model as an input. The utility function will return `input_ids` for text and `inputs_embeds` for speech both of which are natively supported by `model.generate` from the transformers module.

```python
# EITHER get inputs from text
prompt = "Okay, so what would be an example of a healthier breakfast option then. Can you tell me?"
inputs = orpheus.get_inputs(text=prompt)
```



``` python
#OR get inputs from speech
import torchaudio
speech_file = "orpheus/assets/input_speech_0.wav"
waveform, sample_rate = torchaudio.load(SPEECH_WAV_PATH)
inputs = orpheus.get_inputs(speech=y)

#for Jupyter Notebook users listen to the input_speech
import IPython.display as ipd 
ipd.Audio(waveform, rate=sample_rate)
```

The `**inputs` for text are given in the form of `input_ids`, the `**inputs` for speech provided by the utility function are in the form of `inputs_embeds`, both of which are compatible with HuggingFace Transformers.

``` python
output_tokens = model.generate(
    **inputs, 
    max_new_tokens=2000, 
    repetition_penalty=1.1, 
    temperature=0.7
    )

output = orpheus.parse_output_tokens(output_tokens[0])

if(message in output):
    print(f"There was an error: {output["message"]}")
else:
    text_output = output["text"]
    output_waveform = output["speech"]

print(text_output)

# use IPython in a Jupyter environment 
import IPython.display as ipd 
ipd.Audio(output_waveform, rate=sample_rate)

# or save/manipulate the output
import torchaudio
torchaudio.save("model_output.wav", output_waveform, 24000)
```

#### Conversational Inference (multi-turn)

Multiturn Inference is the equivalent of stacking multiple single turn inferences on top of each other. We instead choose to store the existing conversation as embedding vectors, i.e. for transformers inputs_embeds. You can do this manually without too much difficulty, or use the utility class below.

```python
conversation = orpheus.initialise_conversation_model() # initialise a new conversation

# format a message object
speech_file = "input_speech_0.wav"
y, sr = librosa.load(speech_file, sr=16000, mono=True)
first_message = {
    "format":"speech",
    "data": y
}

conversation.append_message(first_message)
text_response_1, waveform_response_1 = conversation.generate_response()

second_message = {
    "format": "text",
    "data": "Where are those foods from?"
}

conversation.append_message(second_message)
text_response_2, waveform_response_2 = conversation.generate_response()
```
