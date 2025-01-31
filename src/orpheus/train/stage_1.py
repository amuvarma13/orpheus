class Stage_1_Trainer():
    def __init__(
            self, 
            model, 
            dataset, text_dataset="amuvarma/5k-qa-pairs-tttts", 
            speech_dataset = "amuvarma/va-320k-330k-snac-no-identity-QA_TTTTS", 
            use_wandb = False,
            wandb_project_name = None,
            wandb_run_name = None,
            model_name = None
        ):
        self.dataset = dataset
        self.model = model
        pass

    def _create_trainer(self):
        pass
            