class OrpheusTrainer():
    def __init__(self):
        self.dataset = None
        pass

    def _load_dataset(self, dataset_name):
        pass

    def _load_model(self, model_name):
        pass

    def initialise (self, 
                    stage="stage_1", 
                    text_dataset_name="amuvarma/stage_1_training_example",
                    speech_dataset_name="amuvarma/stage_1_training_example",
                    dataset_name = None,
                    use_wandb = True,
                    wandb_project_name = None, 
                    wandb_run_name = None, 
                    model_name = None, 
                    base_model_name = None,
                ):
        
        if model_name is not None:
            self._load_model(model_name)
        
        if base_model_name is not None:
            self._load_model(base_model_name)

        if dataset_name is not None:
            self._load_dataset(dataset_name)
        
        if text_dataset_name is not None:
            self._load_dataset(text_dataset_name)
        
        if speech_dataset_name is not None:
            self._load_dataset(speech_dataset_name)
        
        assert stage in ["stage_1", "stage_2", "stage_3", "stage_4"], "Please pass valid stage."

        if stage == "stage_1":
            assert text_dataset_name is not None, "Please pass text_dataset_name."
            assert speech_dataset_name is not None, "Please pass speech_dataset_name."

        if stage == "stage_2":
            assert dataset_name is not None, "Please pass dataset_name."
            assert model_name is not None, "Please pass the name of the model you trained in stage 1."

        if stage == "stage_3":
            assert model_name is not None, "Please pass model_name."
            assert base_model_name is not None, "Please pass the name of the model you trained in stage 1 or stage 2 if you chose to do stage 2."

        if stage == "stage_4":
            assert dataset_name is not None, "Please pass the name of the processed dataset."
            assert model_name is not None, "Please pass model_name you trained in stage 3."
            assert base_model_name is not None, "Please pass the name of the model you trained in stage 1 or stage 2 "
