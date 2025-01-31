import AutoTokenizer
from datasets import load_dataset

class OrpheusDataProcessor():
        def __init__(
            self, 
            tokenizer_name = "amuvarma/3b-10m-pretrain-full",
            text_dataset_name=None,
            speech_dataset_name=None,
            dataset_name = None,
        ):
            self.tokenizer = self._load_tokenizer(tokenizer_name)
            self.text_dataset = self._load_dataset(text_dataset_name)
            self.speech_dataset = self._load_dataset(speech_dataset_name)
            self.dataset = self._load_dataset(dataset_name)
                

        def _load_dataset(self, dataset_name):
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",   
                revision="main",        
                max_workers=64         
            )
            return load_dataset(dataset_name, split="train")
        
        def _load_tokenizer(self, model_name):
            return AutoTokenizer.from_pretrained(model_name)