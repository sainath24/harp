import os
import json

from constants import *

def validate_config(config):
    if "application" not in config.keys():
        print(f"ERROR: Application name not provided in config")
        return False

    return True
    

def create_application_config(application: str):
    pipeline_home_path = os.getenv(PIPELINE_HOME_VAR)

    application_home = os.path.join(pipeline_home_path, APPLICATION_FOLDER_NAME)
    if not os.path.isdir(application_home):
        os.mkdir(application_home)
    
    application_path = os.path.join(application_home, application)
    os.mkdir(application_path)

    train_path = os.path.join(application_path, TRAIN_DATASET_FOLDER_NAME)
    os.mkdir(train_path)

    #models path
    models_path = os.path.join(application_path, MODELS_FOLDER_NAME)
    os.mkdir(models_path)

    config = default_config
    config["application"] = application
    config["train_folder"] = train_path
    config["dataset"] = str(os.path.join(application_path, DATASET_FILE_NAME))
    config["models"]["linear_regression"] = os.path.join(models_path, LINEAR_REGRESSION_MODEL_NAME)
    config["models"]["dnn"] = os.path.join(models_path, DNN_MODEL_NAME)
    config["models"]["decision_tree_regressor"] = os.path.join(models_path, DECISION_TREE_REGRESSOR_MODEL_NAME)

    config_path = os.path.join(application_path, APPLICATION_CONFIG_FILE_NAME)
    with open(config_path, "w") as file:
        json.dump(config, file)

def save_application_config(config):
    application = config["application"]

    pipeline_home_path = os.getenv(PIPELINE_HOME_VAR)
    application_path = os.path.join(pipeline_home_path, APPLICATION_FOLDER_NAME, application)
    config_path = os.path.join(application_path, APPLICATION_CONFIG_FILE_NAME)

    with open(config_path, "w") as file:
        json.dump(config, file)

    

def get_application_config(application: str):
    # GET APPLICATION FOLDER FROM ENV
    pipeline_home_path = os.getenv(PIPELINE_HOME_VAR)
    application_path = os.path.join(pipeline_home_path, APPLICATION_FOLDER_NAME, application)
    if not os.path.isdir(application_path): #directory doesnt exist
        print(f"Application config was not found, creating a config for `{application}`...")
        create_application_config(application)
    
    config_path = os.path.join(application_path, APPLICATION_CONFIG_FILE_NAME)

    with open(config_path, "r") as file:
        config = json.load(file)

    return config


default_config = {
    "application": None,
    "dataset": None,
    "train_folder": None,
    "num_rows": None,
    "last_modified": None,
    "accuracy": {
        "train": None,
        "test": None
    },
    "models": {
        "linear_regression": None,
        "dnn": None,
        "decision_tree_regressor": None
    }
}
