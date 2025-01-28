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
```bash
from mm_model_for_colab import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
)
```
If you are running this normally import the default version
```bash
from mm_model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
)
```


4. Register the model type with transformers

```python
from transformers import AutoModel, AutoTokenizer, AutoConfig
AutoConfig.register("orpheus", OrpheusConfig)
AutoModel.register(OrpheusConfig, OrpheusForConditionalGeneration)
```

5. 