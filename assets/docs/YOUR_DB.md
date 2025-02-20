# Custom Dataset

To support your own dataset in our codebase, complete the following steps:


### 1. Create Dataset Class

Create your dataset class with data loading functionalities and place it in `src/data/`.
This module should load your data on demand.
By default, our datasets return two outputs: a torch tensor with video frames, and a dictionary
with caption information, such as the raw text, text tokens or other metadata.
We recommend sticking to this convention.


### 2. Support Dataset Loading

In the file  `src/data/load_data.py`, modify the `load_data` and `unwrap_batch_data` functions to support
loading your dataset.


### 3. Add your Dataset Config

Add a new configuration file for your dataset in `src/configs/datasets/`.
This is a dictionary that features all the arguments required by your data, as well as 
the default value for each argument.


### 4. Test that your Dataset is Recognized

Completing the previous steps should support your dataset.
You can test it by running: `src/01_create_experiment.py --help`.
If your dataset is correctly added, it should appear in the list of available datasets.

```
...
  --dataset_name DATASET_NAME
            Dataset name to add to the exp_params: ['CATER_Easy', 'CATER_Hard', 'CLIPort', 'YOUR_DB']
...
```

You can now train your our decomposition models and predictors on your own custom dataset.


In case of questions, do not hesitate to open an issue or contact the authors at `villar@ais.uni-bonn.de`
