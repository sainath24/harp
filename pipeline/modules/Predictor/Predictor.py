from PipelineModule import PipelineModule
from .model_selector import model_selector as model_selector

class Predictor(PipelineModule):
    def __init__(self, config: dict):
        super().__init__()
        self.application = config["application"]
        self.config = config

    def get_application_selector_csv(self): 
        #TODO: read app specific csv

        pass

    def write_selector_results(self, results):
        #TODO: write app config and app specific csv
        pass

    def execute(self):
        # TODO pass application csv, get result dataframe, write it and write conclustion to app config file
        if self.application == "DNN":
            selector_dataframe = self.get_application_selector_csv()
            results = model_selector(selector_dataframe)

        
        print(f"ERROR: Invalid application `{self.application}` passed to Predictor.")
        exit()