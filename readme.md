# Inference

### Setup Environment

#### 1. Installation
Clone this repository.
```bash
pip install canopy-orpheus
```

#### 2. Import relevant Orpheus modules

Due to how colab processes modules if you are on Colab import the  correct version.
```python
from orpheus import OrpheusUtility
orpheus = OrpheusUtility()
```

#### 3. Initialise the model

Now we register the model so that we can use it with AutoModel and AutoTokenizer.

```python
import torch
from transformers import AutoModel, AutoTokenizer

orpheus.initialise()

model_name = "amuvarma/zuck-3bregconvo-automodelcompat"
model = AutoModel.from_pretrained(model_name).to("cuda").to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

orpheus.register_auto_model(model=model, tokenizer=tokenizer)
```

### Run Inference

The model can accept both text and speech inputs and outputs both text and speech outputs. You can use this model much like any LLM found on huggingface transformers.

This section will show you how to run inference on text inputs, speech inputs, or multiturn conversations with combined inputs. We use a standard format for chats with ```start_of_human```, ```end_of_human```, ```start_of_ai```, and ```end_of_ai``` tokens to guide the model to understand whose turn it is.

#### Simple Inference (1-turn)

We can pass either text (shown below), speech(shown below), or a combination of text and speech (not shown below) to the model as an input. The utility function will return `input_ids` for text and `inputs_embeds` for speech both of which are natively supported by `model.generate` from the transformers module.

##### Get inputs from text
```python
prompt = "Okay, so what would be an example of a healthier breakfast option then. Can you tell me?"
inputs = orpheus.get_inputs(text=prompt)
```
##### Get inputs from speech

We provide a speech file for you to test out the model quickly as follows:

``` python
import requests
from io import BytesIO
import torchaudio

response = requests.get(orpheus.get_dummy_speech_link()) 
audio_data = BytesIO(response.content)
waveform, sample_rate = torchaudio.load(audio_data) # replace with your own speech

#for Jupyter Notebook users listen to the input_speech
import IPython.display as ipd 
ipd.Audio(waveform, rate=sample_rate)

inputs = orpheus.get_inputs(speech=waveform)
```

The `**inputs` for text are given in the form of `input_ids`, the `**inputs` for speech provided by the utility function are in the form of `inputs_embeds`, both of which are compatible with HuggingFace Transformers.

``` python
with torch.no_grad()
    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=2000, 
        repetition_penalty=1.1, 
        temperature=0.7, 
        eos_token_id=orpheus.special_tokens["end_of_ai"]
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

*NB: The provided model hasn't been finetuned as much towards multiturn dialogues as question answering. Use the appropriate training script to tune the model to your needs.*

##### Initialise a conversation 
``` python
conversation = orpheus.initialise_conversation() # initialise a new conversation
```

We can now pass our inputs to the conversation class.

##### Create a message object
We create a conversation by adding messages to it. Messages follow a similar pattern as shown below regardless if they are text or speech for the input.
``` python
import requests
from io import BytesIO
import torchaudio

response = requests.get(orpheus.get_dummy_speech_link()) 
audio_data = BytesIO(response.content)
waveform, sample_rate = torchaudio.load(audio_data)

first_message = {
    "format":"speech",
    "data": waveform
}

conversation.append_message(first_message)
```

##### Get the response

Depending on how long the output of the model is, and your hardware, this can take up to 2 minutes. We are currently working on providing an implementation of realtime streaming.

``` python
output = conversation.generate_response()

print(output["text"])
ipd.Audio(output["waveform"], rate=24000)
```
##### Multiturn conversation

You can now extend the conversation and all future dialogues will have context of what has been said.

``` python
second_message = {
    "format": "text",
    "data": "Can you give me some ideas for lunch?"
}

conversation.append_message(second_message)
text_response_2, waveform_response_2 = conversation.generate_response()
```

