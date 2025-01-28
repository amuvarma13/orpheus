1. Installation
Clone this repository.
```bash
git clone https://github.com/amuvarma13/orpheus.git
pip install snac
```

2. Additional setup
If you are running this model on Google you can skip this step. Cuda version should be 12.X otherwise you will have conflicts with version of `flash_attn`. These can be resolved by reverting to an earlier version of flash_attn. 
```pip install torch==2.5.1 torchaudio transformers==4.47.0 flash_attn librosa soundfile```

Colab will have appropriate versions of these packages installed.

3. Import relevant modules

```from transformers import AutoModel, AutoTokenizer, ModelRegistry