from PipelineModule import PipelineModule
from .TrimmomaticTrainer import train as trimmomatic_train
from .DNNTrainer import train as dnn_train

import pandas as pd
import os
import pickle
import re


class ModelTrainer(PipelineModule):
    def __init__(self, config: dict, dataset = None):
        super().__init__()
        self.application = str.lower(config["application"])
        self.config = config
        self.dataset = dataset

    def load_dataset(self):
        metadata = self.config["application_config"]
        if not os.path.isfile(metadata["dataset"]):
            print(f"ERROR: ModelTrainer was unable to find a dataset for `{self.application}`")
            exit()
        self.dataset = pd.read_csv(metadata["dataset"])

    def save_models(self, models):
        print("INFO: Saving models...")
        metadata = self.config["application_config"]
        # model_paths = metadata["models"]
        models_home = os.path.join(os.getenv("PIPELINE_HOME"), "applications", self.application,"models")
        for k,v in models.items():
            model_path = os.path.join(models_home, k)
            lr_regex = re.compile('lr_*')
            dtr_regex = re.compile('dtr_*')
            dnn_regex = re.compile('dnn_*')
            if lr_regex.search(k) or dtr_regex.search(k): #save scikit-learn model
                pickle.dump(v, open(model_path, "wb"))
            elif dnn_regex.search(k): # tensorflow model
                v.save(model_path)

    def save_model_dataset(self, dataframe):
        print("INFO: Saving models info...")
        metadata = self.config["application_config"]
        # model_paths = metadata["models"]
        model_dataset_path = os.path.join(os.getenv("PIPELINE_HOME"), "applications", self.application, "models", "models.csv")
        if os.path.isfile(model_dataset_path):
            dataframe.to_csv(model_dataset_path, mode='a', index=False, header=False)
        else:
            dataframe.to_csv(model_dataset_path, header = True, index = False)


    def should_train(self) -> bool: 
        metadata = self.config["ModelTrainer"]
        try:
            should_train = metadata["train"]
            if should_train:
                self.load_dataset()
            return should_train
        except:
            print("Key not found in metadata")
            pass #key not found

        #The below is used when data preprocessor did not set should train(eg: manually updating a dataset)
        #should trian returns true if the change in datasets in >= 15%
        self.load_dataset()
        num_rows = float(self.dataset.shape[0])
        old_num_rows = float(metadata["num_rows"] or 0.01) #to avoid None and division by zero
        change = (num_rows - old_num_rows)/old_num_rows
        if change >= 0.15:
            metadata["num_rows"] = num_rows
            return True

        return False

    def execute(self):
        #TODO: write MAE, MSE, etc values to csv/ maintain a table
        # return models, data from train
        if self.application == "trimmomatic":
            if self.should_train():
                models, pd_df = trimmomatic_train(self.dataset)
                self.save_models(models)
                self.save_model_dataset(pd_df)
            else:
                print(f"Model retraining not required for `{self.application}`")
                return
        elif self.application == "dnn":
            if self.should_train():
                models, pd_df = dnn_train(self.dataset)
                self.save_models(models)
                self.save_model_dataset(pd_df)
            else:
                print(f"Model retraining not required for `{self.application}`")
                return
        else:
            print(f"Unable to train model for invalid application `{self.application}`")
            exit()

        print("INFO: ModelTrainer completed successfully")

        
