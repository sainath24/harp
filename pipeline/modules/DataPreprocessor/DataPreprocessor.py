from PipelineModule import PipelineModule
from .TrimmomaticPreprocessor import preprocess as trimmomatic_preprocess
from .GrayScottPreprocessor import preprocess as gray_scott_preprocess
from .DNNPreprocessor import preprocess as dnn_preprocess

import os


class DataPreprocessor(PipelineModule):
    def __init__(self, config):
        super().__init__()
        self.application = str.lower(config["application"])
        self.config = config


    def set_should_train(self, num_rows):
        metadata = self.config["application_config"]
        old_num_rows = metadata["num_rows"] or 0.01 #to avoid none and division by zero
        metadata["num_rows"] = num_rows
        change = float(num_rows - old_num_rows) / old_num_rows
        if change >= 0.15:
            metadata["train"] = True


    def write_csv(self, dataset):
        application_config = self.config["application_config"]
        dataset.to_csv(application_config["dataset"], header = True, index = False)

    def get_dataset_list(self):
        application_config = self.config["application_config"]
        train_path = application_config["train_folder"]
        dataset_list = []
        for file_name in os.listdir(train_path):
            if file_name[-4:] == ".csv":
                dataset_list.append(os.path.join(train_path, file_name))

        return dataset_list


    def execute(self):
        #TODO get dataset list
        if self.application == "trimmomatic":
            dataset_list = self.get_dataset_list()
            dataset = trimmomatic_preprocess(dataset_list)
            self.write_csv(dataset)

        elif self.application == "dnn":
            dataset_list = self.get_dataset_list()
            dataset = dnn_preprocess(dataset_list)
            self.write_csv(dataset)

        elif self.application == "grayscott":
            dataset_list = self.get_dataset_list()
            dataset = gray_scott_preprocess(dataset_list)
            self.write_csv(dataset)

        else:
            print(f"Invalid application `{self.application}` passed to data preprocessor")

        print("INFO: DataPreprocessor completed successfully")

        return dataset
