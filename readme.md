### SETUP
1. Installation
Clone this repository.
```bash
git clone https://github.com/amuvarma13/orpheus.git
pip install snac
```


2. Additional setup
If you are running this model on Google Colab you can skip this step. Cuda version should be 12.X otherwise you will have conflicts with the version of `flash_attn`. These can be resolved by reverting to an earlier version of `flash_attn`. 
```bash
pip install torch==2.5.1 torchaudio transformers==4.47.0 flash_attn librosa soundfile
```

Colab will have appropriate versions of these packages installed.


3. Import relevant Orpheus modules

Due to how colab processes modules if you are on Colab import the  correct version.
```python
from orpheus.mm_model_from_colab.model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
)
```
If you are running this normally import the default version
```python
from mm_model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
    fast_download_from_hub
)
```


4. Register the model type with transformers

Now we register the model so that we can use it with AutoModel and AutoTokenizer.

```python
from transformers import AutoModel, AutoTokenizer, AutoConfig
AutoConfig.register("orpheus", OrpheusConfig)
AutoModel.register(OrpheusConfig, OrpheusForConditionalGeneration)
```

5. Compose Model

We have create helper functions you can use to compose the model faster

```python
compose_model() # downloads relevant folders from hub in ~2-3 minutes
```

<!-- We now create the 
```python
text_model_id= "amuvarma/3b-zuckreg-convo"
mm_model_id = "amuvarma/3b-zuckreg-convo-projsnactune"

config = OrpheusConfig(
    text_model_id=model_id,
    audio_token_index=156939,
    vocab_size=156939,
)
orpheus = AutoModel.from_pretrained(mm_model_id, config=config, new_vocab_size=False).to(dtype=torch.bfloat16).to("cuda")
```
 -->


