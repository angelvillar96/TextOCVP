"""
Methods to manage parameters and configurations
"""

import os
import json

from configs import get_config
from lib.logger import print_
from lib.utils import timestamp
import configs
from CONFIG import DEFAULTS



class Config(dict):
    """
    Main module to initialize, save, load, and process the experiment parameters.
    """
    _default_values = DEFAULTS
    _config_groups = ["dataset", "model", "training", "loss"]

    def __init__(self, exp_path):
        """
        Populating the dictionary with the default values
        """
        for key in self._default_values.keys():
            self[key] = self._default_values[key]
        self["_general"] = {}
        self["_general"]["exp_path"] = exp_path
        return

    def create_exp_config_file(self, model_name, dataset_name, exp_path=None):
        """
        Creating a JSON file with exp configs in the experiment path
        """
        exp_path = exp_path if exp_path is not None else self["_general"]["exp_path"]
        if not os.path.exists(exp_path):
            raise FileNotFoundError(f"ERROR!: exp_path {exp_path} does not exist...")

        # adding to exp-params the parameters from the specified dataset and model
        for key in Config._default_values.keys():
            if key == "model":
                self["model"] = configs.get_model_config(model_name)
            elif key == "dataset":
                self["dataset"] = configs.get_dataset_config(dataset_name)
            elif key in ["prediction_params", "prediction_params", "predictor_loss"]:
                _ = self.pop(key)
                continue
            else:
                self[key] = Config._default_values[key]

        # updating general and saving
        self["_general"]["created_time"] = timestamp()
        self["_general"]["last_loaded"] = timestamp()
        exp_config = os.path.join(exp_path, "experiment_params.json")
        with open(exp_config, "w") as file:
            json.dump(self, file)
        return

    def load_exp_config_file(self, exp_path=None):
        """
        Loading the JSON file with exp configs
        """
        if exp_path is not None:
            self["_general"]["exp_path"] = exp_path
        exp_config = os.path.join(
                self["_general"]["exp_path"],
                "experiment_params.json"
            )
        if not os.path.exists(exp_config):
            raise FileNotFoundError(f"ERROR! {exp_config = } does not exist...")

        with open(exp_config) as file:
            self = json.load(file)
        self["_general"]["last_loaded"] = timestamp()
        return self

    def save_exp_config_file(self, exp_path=None, exp_params=None):
        """
        Dumping experiment parameters into path
        """
        exp_path = self["_general"]["exp_path"] if exp_path is None else exp_path
        exp_params = self if exp_params is None else exp_params

        exp_config = os.path.join(exp_path, "experiment_params.json")
        with open(exp_config, "w") as file:
            json.dump(exp_params, file)
        return


    def add_predictor_parameters(self, exp_params, predictor_name):
        """
        Adding predictor parameters and predictor training meta-parameters
        to exp_params dictionary, and removing some unnecessaty keys
        """
        # adding predictor parameters
        predictor_params = get_config(key="predictors", name=predictor_name)
        exp_params["predictor"] = predictor_params

        # adding predictor training and predictor loss parameteres
        exp_params["prediction_params"] = DEFAULTS["prediction_params"]
        exp_params["predictor_loss"] = DEFAULTS["predictor_loss"]

        # reodering exp-params to have the desired key orderign
        sorted_keys = [
                "dataset", "model", "predictor", "predictor_loss",
                "training", "prediction_params", "_general"
            ]
        exp_params = {k: exp_params[k] for k in sorted_keys}
        return exp_params


