"""
Global configurations
"""

import os


CONFIG = {
    "random_seed": 14,
    "epsilon_min": 1e-16,
    "epsilon_max": 1e16,
    "num_workers": 8,
    "paths": {
        "data_path": os.path.join(os.getcwd(), "/home/nfs/inf6/data/datasets"),
        "experiments_path": os.path.join(os.getcwd(), "experiments"),
        "configs_path": os.path.join(os.getcwd(), "src", "configs"),
    }
}



DEFAULTS = {
    "dataset": {
        "dataset_name": "",
        "shuffle_train": True,
        "shuffle_eval": False,
    },
    "model": {
        "model_name": "",
        "model_params": {}
    },
    "predictor":{
        "predictor_name": "",
        "predictor_params": {}
    },
    "loss": [
        {
            "type": "mse",
            "weight": 1
        }
    ],
    "predictor_loss": [
        {
          "type": "pred_img_mse",
          "weight": 1
        },
        {
          "type": "pred_slot_mse",
          "weight": 1
        }
    ],
    "training": {
        "num_epochs": 1000,        
        "save_frequency": 25,
        "log_frequency": 100,
        "image_log_frequency": 300,
        "batch_size": 64,
        "lr": 1e-4,
        "scheduler": "cosine_annealing",
        "scheduler_steps": 1e6,
        "lr_warmup": True,
        "warmup_steps": 2000,
        "gradient_clipping": True,
        "clipping_max_value": 0.05
    },
    "prediction_params": {
        "num_context": 1,
        "num_preds": 9,
        "teacher_force": False,
        "input_buffer_size": 10,
    }
}



COLORS = ["white", "blue", "green", "olive", "red", "yellow", "purple", "orange", "cyan",
          "brown", "pink", "darkorange", "goldenrod", "darkviolet", "springgreen",
          "aqua", "royalblue", "navy", "forestgreen", "plum", "magenta", "slategray",
          "maroon", "gold", "peachpuff", "silver", "aquamarine", "indianred", "greenyellow",
          "darkcyan", "sandybrown"]


#
