# Training

We provide our entire pipeline for training TextOCVP.
This process happens in two different stages:
 1. Training an object-centric video decomposition, e.g., SAVi or ExtendedDINOSAUR.
 2. Training TextOCVP for text-guided object-centric image to video generation.



## 1. Train Object-Cetric Video Decomposition Model

**1.** Create a new experiment using the `src/01_create_experiment.py` script.
       This will create a new experiments folder in the `/experiments` directory.

```
usage: 01_create_experiment.py [-h] -d EXP_DIRECTORY --name NAME [--model_name MODEL_NAME] [--dataset_name DATASET_NAME]

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Directory where the experiment folder will be created
  --name NAME           Name to give to the experiment
  --model_name MODEL_NAME
                        Model name to add to the exp_params: ['ExtendedDINOSAUR', 'SAVi']
  --dataset_name DATASET_NAME
                        Dataset name to add to the exp_params: ['CATER_Easy', 'CATER_Hard', 'CLIPort']
```



**2.** Modify the experiment parameters located in `experiments/YOUR_EXP_DIR/YOUR_EXP_NAME/experiment_params.json` to adapt to your dataset and training needs.



**3.** Train the object-centric video decomposition model given the specified experiment parameters.
This can be done with either the `src/02_train_savi.py` or `src/02_train_extended_dinosaur.py` scripts:

```
usage: 02_train_savi.py [-h] -d EXP_DIRECTORY [--checkpoint CHECKPOINT] [--resume_training]

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Directory where the experiment folder will be created
  --checkpoint CHECKPOINT
                        Checkpoint with pretrained parameters to load
  --resume_training     For resuming training
```


You can visualize the training progress in the Tensorboard.
```
tensorboard --logdir experiments/YOUR_EXP_DIR/YOUR_EXP_NAME/ --port TBOARD_PORT
```



#### Example: SAVi Training

Below we provide an example of how to train a new SAVi model for object-centric video decomposition on the CATER dataset.:

```
python src/01_create_experiment.py \
    -d experiments \
    --name my_exp_on_CATER \
    --model_name SAVi
    --dataset_name CATER_Hard

python src/02_train_savi.py \
    -d experiments/my_exp_on_CATER
```




## 2. Training the Text.-Guided Object-Centric Predictor Model

Training an object-centric video prediction model requires having a pretrained object-centric video decomposition model.
You can use either our provided pretrained models, or you can train your own video decomposition models.


**1.** Create a new predictor experiment using the `src/src/01_create_predictor_experiment.py` script. This will create a new predictor folder in the specified experiment directory.

```
usage: 01_create_predictor_experiment.py [-h] -d EXP_DIRECTORY --name NAME --predictor_name {OCVPSeq,TextOCVP_CustomTF,TextOCVP_T5,VanillaTransformer}

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Obj-Decomp exp. directory where the predictor exp. will be created
  --name NAME           Name to give to the predictor experiment
  --predictor_name {OCVPSeq,TextOCVP_CustomTF,TextOCVP_T5,VanillaTransformer}
                        Name of the predictor module to use: ['OCVPSeq', 'TextOCVP_CustomTF', 'TextOCVP_T5', 'VanillaTransformer']
```



**2.** Modify the experiment parameters located in `experiments/YOUR_EXP_DIR/YOUR_EXP_NAME/predictors/YOUR_PREDICTOR_NAME/experiment_params.json` to adapt the predictor training parameters to your training needs




 **3.** Train the Text-Guided Object-Centric Predictor TextOCVP module given the specified experiment parameters and a pretrained decomposition model:

```
usage: 04_train_predictor.py [-h] -d EXP_DIRECTORY --decomp_ckpt DECOMP_CKPT --name_pred_exp NAME_PRED_EXP [--checkpoint CHECKPOINT] [--resume_training]

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Path to the experiment directory
  --decomp_ckpt DECOMP_CKPT
                        Checkpoint with pretrained parameters to load
  --name_pred_exp NAME_PRED_EXP
                        Name to the predictor experiment directory
  --checkpoint CHECKPOINT
                        Checkpoint with pretrained parameters to load
  --resume_training     For resuming training
```


#### Example: Predictor Training

Below we provide an example of how to train an object-centric predictor given a pretrained SAVi model. This example continues the example above

```
python src/01_create_predictor_experiment.py \
  -d experiments/my_exp_on_CATER \
  --name my_TextOCVP_exp \
  --predictor_name TextOCVP_T5

python src/04_train_predictor.py \
  -d experiments/my_exp_on_CATER
  --decomp_ckpt checkpoint_epoch_final.pth
  --name_pred_exp my_TextOCVP_exp
```




## Further Comments

 - You can download our experiments directory, including the experiment parameters and pretrained 
 checkpoints by running the following script:
  ```
  chmod +x download_pretrained.sh
  ./download_pretrained.sh
  ```


 - The training can be monitored using Tensorboard.
   To launch tensorboard,
  
  ```
  tensorboard --logdir experiments/EXP_DIR/EXP_NAME --port 8888
  ```

 - In case of questions, do not hesitate to open an issue or contact the authors at `villar@ais.uni-bonn.de`
