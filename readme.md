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


##### Get inputs from speech

We provide a speech file for you to test out the model quickly as follows. There is an example of how to pass text inputs into the model below.

``` python
import requests
from io import BytesIO
import torchaudio

response = requests.get(orpheus.dummy_speech_link) 
audio_data = BytesIO(response.content)
waveform, sample_rate = torchaudio.load(audio_data) # replace with your own speech

#for Jupyter Notebook users listen to the input_speech
import IPython.display as ipd 
ipd.Audio(waveform, rate=sample_rate)

inputs = orpheus.get_inputs(speech=waveform)
```

##### Call model.generate
The `**inputs` for text are given in the form of `input_ids`, the `**inputs` for speech provided by the utility function are in the form of `inputs_embeds`, both of which are compatible with HuggingFace Transformers.

``` python
with torch.no_grad():
    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=2000, 
        repetition_penalty=1.1, 
        temperature=0.7, 
        eos_token_id=orpheus.special_tokens["end_of_ai"]
    )

output = orpheus.parse_output_tokens(output_tokens)

if output["message"] is not None:
    print(f"There was an error: {output['message']}")
else:
    text_output = output["text"]
    output_waveform = output["speech"]

print(text_output)

# use IPython in a Jupyter environment 
import IPython.display as ipd 
ipd.Audio(output_waveform, rate=24000)

# or save/manipulate the output
from scipy.io import wavfile
wavfile.write("output.wav", 24000, output_waveform)
```

##### Get inputs from text

You can create `**inputs` from text as shown below. You call `model.generate` and parse the output tokens exactly as described above with speech.

```python
prompt = "Okay, so what would be an example of a healthier breakfast option then. Can you tell me?"
inputs = orpheus.get_inputs(text=prompt)
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

message_0 = {
    "format":"speech",
    "data": waveform
}

conversation.append_message(message_0)
```

##### Get the response

Depending on how long the output of the model is, and your hardware, this can take up to 2 minutes. We are currently working on providing an implementation of realtime streaming.

``` python
output_0 = conversation.generate_response()

print(output_0["text"])
ipd.Audio(output_0["speech"], rate=24000)
```
##### Multiturn conversation

You can now extend the conversation and all future dialogues will have context of what has been said.

``` python
message_1 = {
    "format": "text",
    "data": "Can you give me some ideas for lunch?"
}

conversation.append_message(message_1)
output_1 = conversation.generate_response()
print(output_1["text"])
ipd.Audio(output_1["speech"], rate=24000)
```

### Inference FAQS
<details>
  <summary><strong>Why is the speech is getting cut off?</strong></summary>
  <p></p>
  <p>The model generates speech autogressively, which means that if the model terminates generation because it has hit the max_tokens criterion it will not finish generating the entire speech sample. You need to increase max_tokens to get the full generation.</p>
</details>

<details>
  <summary><strong>How many seconds of speech can I generate per inference? </strong></summary>
  <p></p>
  <p>While there is no limit on how many seconds of speech the model can respond with, the model has been mostly trained on sequences less than a 60 seconds. Each second of speech generated requires 83 tokens. </p>
</details>

<details>
  <summary><strong>How do I run inference in realtime? </strong></summary>
  <p></p>
  <p>Using an inference optimised library like vllm will allows you to run Orpheus in realtime. We are working on an implementation.</p>
</details>

<details>
  <summary><strong>I want to customise the model can I prompt it? </strong></summary>
  <p></p>
  <p>Currently the best way to customise the model (and how we want developers to customise) is by finetuning it. This should be very simple with the scripts provided. The reason for this is because we want to explore better ways of post training. </p>
</details>

<details>
  <summary><strong>What are the strengths/limitations of this model? </strong></summary>
  <p></p>
  <p>While we have extended the training of Llama-3b on large amounts of speech and text data, there are limitations. The model is not good at niche words, numbers in numerical form, and proper nouns. It is also a very small model so it lacks textual based reasoning and knowledge (especially after it forgets some of this when trained on speech).
  
  Since this model is small it is cheaper to finetune and we provide very simple scripts to add a high degree of customisability to the voice, emotions, intonations, personality, and knowledge of the model.
  
  We will also soon release a bigger, more extensively trained model that doesn't have any of the above issues.</p>
</details>