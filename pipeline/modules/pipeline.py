
from DataPreprocessor.DataPreprocessor import DataPreprocessor
from DataScraper.DataScraper import DataScraper
from ModelTrainer.ModelTrainer import ModelTrainer
from Predictor.Predictor import Predictor
from config_tools import get_application_config, validate_config, save_application_config

import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str, help="path to json config")
    parser.add_argument('--module', default=None, type=str, help="[optional] module to run")
    args = parser.parse_args()
    return args


def launch_data_preprocessor(config: dict):
    module = DataPreprocessor(config)
    dataset = module.execute()
    return dataset

def launch_data_scraper(config: dict):
    module = DataScraper(config)
    module.execute()

def launch_model_trainer(config: dict):
    module = ModelTrainer(config)
    module.execute()

def launch_predictor(config: dict):
    module = Predictor(config)
    module.execute()

def run_all(config: dict):
    launch_data_scraper(config)
    launch_data_preprocessor(config)
    launch_model_trainer(config)
    launch_predictor(config)

def main(module: str, config: dict):
    if module is None:
        run_all(config)

    if module == "data_preprocessor":
        launch_data_preprocessor(config)

    elif module == "data_scraper":
        launch_data_scraper(config)

    elif module == "model_trainer":
        launch_model_trainer(config)

    elif module == "predictor":
        launch_predictor(config)

    else:
        print(f'ERROR: Tried to launch invalid pipeline module `{module}`')


if __name__ == "__main__":
    #TODO allows modules
    args = parse_args()
    module, config_path = args.module, args.config
    if os.path.isfile(config_path) != True:
        print(f"ERROR: invalid path to pipeline config")

    with open(config_path, 'r') as file:
        config = json.load(file)

    #validate config
    is_valid = validate_config(config)
    if is_valid != True: 
        print("ERROR: Invalid pipeline configuration")
        exit()
    
    application_config = get_application_config(config["application"])

    config["application_config"] = application_config #add application config to config

    main(module, config)

    save_application_config(config["application_config"])
