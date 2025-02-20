"""
Methods for processing command line arguments
"""

import os
import argparse

from lib.utils import split_path
from CONFIG import CONFIG


########################################
## ARGUMENTS FOR CREATING EXPERIMENTS ##
########################################


def create_experiment_arguments():
    """
    Processing arguments for 01_create_experiment.py
    """
    from configs import get_available_configs
    MODELS = get_available_configs("models")
    DATASETS = get_available_configs("datasets")

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Directory where the experiment folder will be created",
            required=True,
        )
    parser.add_argument(
            "--name",
            help="Name to give to the experiment",
            required=True
        )
    parser.add_argument(
            "--model_name",
            help=f"Model name to add to the exp_params: {MODELS}"
        )
    parser.add_argument(
            "--dataset_name",
            help=f"Dataset name to add to the exp_params: {DATASETS}"
        )
    args = parser.parse_args()
    args.exp_directory = process_experiment_directory_argument(
            exp_directory=args.exp_directory,
            create=True
        )
    return args


def create_predictor_experiment_arguments():
    """
    Processing arguments for 01_create_predictor_experiment.py
    """
    from configs import get_available_configs
    PREDICTORS = get_available_configs("predictors")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Obj-Decomp exp. directory where the predictor exp. will be created",
            required=True
        )
    parser.add_argument(
            "--name",
            help="Name to give to the predictor experiment",
            required=True
        )
    parser.add_argument(
            "--predictor_name",
            help=f"Name of the predictor module to use: {PREDICTORS}",
            required=True,
            choices=PREDICTORS
        )
    args = parser.parse_args()
    args.exp_directory = process_experiment_directory_argument(
            exp_directory=args.exp_directory,
            create=True
        )
    if args.predictor_name not in PREDICTORS:
        raise ValueError(f"{args.predictor_name = } not in allowed {PREDICTORS = }...")
    return args



#################################################
## ARGUMENTS FOR OBJECT-CENTRIC DECOMOPOSITION ##
#################################################


def get_train_decomp_arguments():
    """
    Processing arguments for main scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Directory where the experiment folder will be created",
            required=True,
        )
    parser.add_argument(
            "--checkpoint",
            help="Checkpoint with pretrained parameters to load",
            default=None
        )
    parser.add_argument(
            "--resume_training",
            help="For resuming training",
            default=False, action='store_true'
        )
    args = parser.parse_args()

    args.exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.checkpoint = process_checkpoint(args.exp_directory, args.checkpoint)
    return args


def get_decomp_eval_arguments():
    """
    Processing arguments for main scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the experiment directory",
            required=True
        )
    parser.add_argument(
            "--decomp_ckpt",
            help="Checkpoint with pretrained parameters to load",
            required=True
        )
    parser.add_argument(
            "--results_name",
            help="Name to give to the results file",
            type=str, required=True
        )
    parser.add_argument(
            "--batch_size",
            help="Overriding the batch size from exp. params for Eval. and Fig-Gen.",
            type=int, default=0
        )
    args = parser.parse_args()

    exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.decomp_ckpt = process_checkpoint(exp_directory, args.decomp_ckpt)
    return exp_directory, args



def get_generate_figs_decomp_model_arguments():
    """
    Processing arguments for predictor evaluation script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the experiment directory",
            required=True
        )
    parser.add_argument(
            "--decomp_ckpt",
            help="Checkpoint with pretrained parameters to load",
            required=True
        )
    parser.add_argument(
            "--num_seqs",
            help="Number of sequences to generate",
            type=int, default=10
        )
    args = parser.parse_args()
    exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.decomp_ckpt = process_checkpoint(exp_directory, args.decomp_ckpt)
    return exp_directory, args



###################################################
## ARGUMENTS FOR OBJECT-CENTRIC VIDEO PREDICTION ##
###################################################


def get_predictor_training_arguments():
    """
    Processing arguments for predictor training script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the experiment directory",
            required=True
        )
    parser.add_argument(
            "--decomp_ckpt",
            help="Checkpoint with pretrained parameters to load",
            required=True
        )
    parser.add_argument(
            "--name_pred_exp",
            help="Name to the predictor experiment directory",
            required=True
        )
    parser.add_argument(
            "--checkpoint",
            help="Checkpoint with pretrained parameters to load",
            default=None
        )
    parser.add_argument(
            "--resume_training",
            help="For resuming training",
            default=False, action='store_true'
        )
    args = parser.parse_args()

    # sanity checks on command line arguments
    args.exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.decomp_ckpt = process_checkpoint(args.exp_directory, args.decomp_ckpt)
    args.name_pred_exp = process_predictor_experiment(
            exp_directory=args.exp_directory,
            name_pred_exp=args.name_pred_exp,    
        )
    args.checkpoint = process_predictor_checkpoint(
            exp_path=args.exp_directory,
            name_pred_exp=args.name_pred_exp,
            checkpoint=args.checkpoint
        )
    return args


def get_predictor_evaluation_arguments():
    """
    Processing arguments for predictor evaluation script.
    """
    parser = argparse.ArgumentParser()
    # base arguments
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the Decomposition Model experiment directory",
            required=True
        )
    parser.add_argument(
            "--decomp_ckpt",
            help="Name of the Decomposition Model checkpoint to use",
            required=True
        )
    parser.add_argument(
            "--name_pred_exp",
            help="Name to the predictor experiment to evaluate.",
            required=True
        )
    parser.add_argument(
            "--pred_ckpt",
            help="Name of the predictor checkpoint to evaluate",
            required=True
        )
    parser.add_argument(
            "-o", "--results_name",
            help="Name to give to the results file",
            type=str, required=True
        )
    # additional arguments
    parser.add_argument(
            "--batch_size",
            help="If provided, it overrides the batch size used for evaluation",
            type=int, default=0
        )
    parser.add_argument(
            "--num_seed",
            help="If provided, it overrides the number of seed frames to use",
            type=int, default=None
        )
    parser.add_argument(
            "--num_preds",
            help="If provided, it overrides the number of frames to predict for",
            type=int, default=None
        )
    args = parser.parse_args()

    # sanity checks on command line arguments
    args.exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.decomp_ckpt = process_checkpoint(args.exp_directory, args.decomp_ckpt)
    args.name_pred_exp = process_predictor_experiment(
            exp_directory=args.exp_directory,
            name_pred_exp=args.name_pred_exp,    
        )
    args.pred_ckpt = process_predictor_checkpoint(
            exp_path=args.exp_directory,
            name_pred_exp=args.name_pred_exp,
            checkpoint=args.pred_ckpt
        )
    if args.batch_size < 1:
        args.batch_size = None
    return args



def get_generate_figs_pred():
    """
    Processing arguments for predictor evaluation script.
    """
    parser = argparse.ArgumentParser()
    # base arguments
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the Decomposition Model experiment directory",
            required=True
        )
    parser.add_argument(
            "--decomp_ckpt",
            help="Name of the Decomposition Model checkpoint to use",
            required=True
        )
    parser.add_argument(
            "--name_pred_exp",
            help="Name to the predictor experiment to evaluate.",
            required=True
        )
    parser.add_argument(
            "--pred_ckpt",
            help="Name of the predictor checkpoint to evaluate",
            required=True
        )
    # prediction arguments
    parser.add_argument(
            "--num_preds",
            help="Number of rollout frames to predict for",
            type=int, default=15
        )
    parser.add_argument(
            "--num_seqs",
            help="Number of sequences to generate",
            type=int, default=30
        )
    args = parser.parse_args()
    
    # sanity checks on command line arguments
    args.exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.decomp_ckpt = process_checkpoint(args.exp_directory, args.decomp_ckpt)
    args.name_pred_exp = process_predictor_experiment(
            exp_directory=args.exp_directory,
            name_pred_exp=args.name_pred_exp,    
        )
    args.pred_ckpt = process_predictor_checkpoint(
            exp_path=args.exp_directory,
            name_pred_exp=args.name_pred_exp,
            checkpoint=args.pred_ckpt
        )
    return args



###############################
## ARGPARSE PROCESSING UTILS ##
###############################


def process_experiment_directory_argument(exp_directory, create=False):
    """
    Ensuring that the experiment directory argument exists
    and giving the full path if relative was detected
    """
    was_relative = False
    exp_path = CONFIG["paths"]["experiments_path"]
    split_exp_dir = split_path(exp_directory)
    if os.path.basename(exp_path) == split_exp_dir[0]:
        exp_directory = "/".join(split_exp_dir[1:])

    if(exp_path not in exp_directory):
        was_relative = True
        exp_directory = os.path.join(exp_path, exp_directory)

    # making sure experiment directory exists
    if(not os.path.exists(exp_directory) and create is False):
        print(f"ERROR! Experiment directorty {exp_directory} does not exist...")
        print(f"     The given path was: {exp_directory}")
        if(was_relative):
            print(f"     It was a relative path. The absolute would be: {exp_directory}")
        print("\n\n")
        exit()
    elif(not os.path.exists(exp_directory) and create is True):
        os.makedirs(exp_directory)

    return exp_directory



def process_predictor_experiment(exp_directory, name_pred_exp):
    """
    If the 'exp_directory' is contained in 'name_pred_exp', we remove the 
    former from the latter.
    """
    if exp_directory in name_pred_exp:
        name_pred_exp = name_pred_exp[len(exp_directory)+1:]
    dirname = "predictors"
    if not name_pred_exp.startswith(f"{dirname}/"):
        name_pred_exp = f"{dirname}/{name_pred_exp}"
    pred_exp_path = os.path.join(exp_directory, name_pred_exp)
    if not os.path.exists(pred_exp_path):
        raise FileNotFoundError(f"{pred_exp_path = } does not exist...")
    return name_pred_exp



def process_checkpoint(exp_path, checkpoint):
    """
    Making sure checkpoint exists
    """
    if checkpoint is not None:
        ckpt_path = os.path.join(exp_path, "models", checkpoint)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{checkpoint = } doesnt exist in {ckpt_path}")
    return checkpoint


def process_predictor_checkpoint(exp_path, name_pred_exp,  checkpoint):
    """
    Making sure predictor checkpoint exists
    """
    if checkpoint is not None:
        ckpt_path = os.path.join(exp_path, name_pred_exp, "models", checkpoint)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{checkpoint = } doesnt exist in {ckpt_path}")
    return checkpoint


#
