"""
Setting up the model, optimizers, loss functions, loading/saving parameters, ...
"""

import os
import traceback
import torch

from configs import get_available_configs
from lib.logger import print_, log_function
from lib.schedulers import LRWarmUp, IdentityScheduler
from lib.utils import create_directory



###########################
## MODEL FACTORY METHODS ##
###########################


@log_function
def setup_model(model_params):
    """
    Loading the model given the model parameters stated in the exp_params file

    Args:
    -----
    model_params: dictionary
        model parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    model: torch.nn.Module
        instanciated model given the parameters
    """
    model_name = model_params["model_name"]
    model_params = model_params["model_params"]
    MODELS = get_available_configs("models")
    if model_name not in MODELS:
        raise NameError(f"'{model_name = }' not in {MODELS = }")

    if(model_name == "SAVi"):
        from models.SAVi import SAVi
        model = SAVi(**model_params)
    elif(model_name == "ExtendedDINOSAUR"):
        from models.ExtendedDINOSAUR import ExtendedDINOSAUR
        model = ExtendedDINOSAUR(**model_params)
    else:
        raise NotImplementedError(
                f"'{model_name = }' not in supported models: {MODELS}"
            )

    return model



@log_function
def setup_predictor(exp_params):
    """
    Loading the predictor given the predictor params. listed in the exp_params file

    Args:
    -----
    predictor_params: dictionary
        model parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    predictor: PredictorWrapper
        Instanciated predictor given the parameters, wrapped into a PredictorWrapper
        to forecast slots for future time steps.
    """
    # model and predictor params
    model_params = exp_params["model"]["model_params"]
    prediction_params = exp_params["prediction_params"]
    predictor_name = exp_params["predictor"]["predictor_name"]
    predictor_params = exp_params["predictor"]["predictor_params"]
    PREDICTORS = get_available_configs("predictors")
    if predictor_name not in PREDICTORS:
        raise NameError(f"Predictor '{predictor_name}' not in {PREDICTORS = }")

    # instanciating predictor
    # OCVP PREDICTORS
    if(predictor_name == "VanillaTransformer"):
        from models.Predictors.OCVP import VanillaTransformerPredictor
        predictor = VanillaTransformerPredictor(
                num_slots=model_params["num_slots"],
                slot_dim=model_params["slot_dim"],
                input_buffer_size=prediction_params["input_buffer_size"],
                **predictor_params
            )
    elif(predictor_name == "OCVPSeq"):
        from models.Predictors.OCVP import OCVPSeq
        predictor = OCVPSeq(
                num_slots=model_params["num_slots"],
                slot_dim=model_params["slot_dim"],
                input_buffer_size=prediction_params["input_buffer_size"],
                **predictor_params
            )
    # TextOCVP PREDICTORS
    elif(predictor_name == "TextOCVP_CustomTF"):
        from models.Predictors.text_cond_OCVP import TextOCVP_CustomTF
        input_buffer_size = prediction_params["input_buffer_size"]
        predictor_params["predictor_params"]["input_buffer_size"] = input_buffer_size
        predictor = TextOCVP_CustomTF(
                slot_dim=model_params["slot_dim"],
                predictor_params=predictor_params.get("predictor_params"),
                fusion_params=predictor_params.get("fusion_params"),
                text_encoder_params=predictor_params.get("text_encoder_params")
            )
    elif(predictor_name == "TextOCVP_T5"):
        from models.Predictors.text_cond_OCVP import TextOCVP_T5
        input_buffer_size = prediction_params["input_buffer_size"]
        predictor_params["predictor_params"]["input_buffer_size"] = input_buffer_size
        predictor = TextOCVP_T5(
                slot_dim=model_params["slot_dim"],
                predictor_params=predictor_params.get("predictor_params"),
                fusion_params=predictor_params.get("fusion_params"),
                text_encoder_params=predictor_params.get("text_encoder_params")
            )
    else:
        raise NameError(
                f"Predictor '{predictor_name}' not in recognized {PREDICTORS = }"
            )

    # instanciating predictor wrapper module to iterate over the data
    from models.Predictors.predictor_wrapper import PredictorWrapper
    predictor = PredictorWrapper(
            exp_params=exp_params,
            predictor=predictor
        )
    return predictor



########################
## SAVING AND LOADING ##
########################


@log_function
def save_checkpoint(model, optimizer, scheduler, lr_warmup, epoch, exp_path,
                    finished=False, savedir="models", savename=None):
    """
    Saving a checkpoint in the models directory of the experiment. This checkpoint
    contains state_dicts for the mode, optimizer and lr_scheduler

    Args:
    -----
    model: torch Module
        model to be saved to a .pth file
    optimizer, scheduler: torch Optim
        modules corresponding to the parameter optimizer and lr-scheduler
    scheduler: Object
        Learning rate scheduler to save
    lr_warmup: Object
        Module performing learning rate warm-up
    epoch: integer
        current epoch number
    exp_path: string
        path to the root directory of the experiment
    finished: boolean
        if True, current checkpoint corresponds to the finally trained model
    """

    if(savename is not None):
        checkpoint_name = savename
    elif(savename is None and finished is True):
        checkpoint_name = "checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"

    create_directory(exp_path, savedir)
    savepath = os.path.join(exp_path, savedir, checkpoint_name)

    scheduler_data = "" if scheduler is None else scheduler.state_dict()
    lr_warmup_data = "" if lr_warmup is None else lr_warmup.state_dict()
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scheduler_state_dict": scheduler_data,
            "lr_warmup": lr_warmup_data
        }, savepath)

    return


@log_function
def load_checkpoint(checkpoint_path, model, only_model=False, map_cpu=False, **kwargs):
    """
    Loading a precomputed checkpoint: state_dicts for the mode, optimizer and lr_scheduler

    Args:
    -----
    checkpoint_path: string
        path to the .pth file containing the state dicts
    model: torch Module
        model for which the parameters are loaded
    only_model: boolean
        if True, only model state dictionary is loaded
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist ...")
    if(checkpoint_path is None):
        return model

    # loading model to either cpu or cpu
    if(map_cpu):
        checkpoint = torch.load(checkpoint_path,  map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path)

    # wrapping predictor into PredictorWrapper for backwards compatibility of ckpts
    first_key_model = list(model.state_dict().keys())[0]
    first_key_checkpoint = list(checkpoint['model_state_dict'].keys())[0]
    if first_key_model.startswith("predictor") and \
       not first_key_checkpoint.startswith("predictor"):
        checkpoint['model_state_dict'] = {
                f"predictor.{key}": val for key, val in checkpoint['model_state_dict'].items()
            }

    # loading model parameters.
    model.load_state_dict(checkpoint['model_state_dict'])

    if(only_model):
        return model

    # returning all other necessary objects
    optimizer = kwargs["optimizer"]
    scheduler = kwargs["scheduler"]
    lr_warmup =  kwargs["lr_warmup"]
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if "lr_warmup" in checkpoint:
        lr_warmup.load_state_dict(checkpoint['lr_warmup'])
    epoch = checkpoint["epoch"] + 1

    return model, optimizer, scheduler, lr_warmup, epoch



def emergency_save(f):
    """
    Decorator for saving a model in case of exception, either from code or triggered.
    Use for decorating the training loop:
        @setup_model.emergency_save
        def train_loop(self):
    """

    def try_call_except(*args, **kwargs):
        """ Wrapping function and saving checkpoint in case of exception """
        try:
            return f(*args, **kwargs)
        except (Exception, KeyboardInterrupt):
            print_("There has been an exception. Saving emergency checkpoint...")
            self_ = args[0]
            if hasattr(self_, "model") and hasattr(self_, "optimizer"):
                fname = f"emergency_checkpoint_epoch_{self_.epoch}.pth"
                save_checkpoint(
                    model=self_.model,
                    optimizer=self_.optimizer,
                    scheduler=self_.warmup_scheduler.scheduler,
                    lr_warmup=self_.warmup_scheduler.lr_warmup,
                    epoch=self_.epoch,
                    exp_path=self_.exp_path,
                    savedir="models",
                    savename=fname
                )
                print_(f"  --> Saved emergency checkpoint {fname}")
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()

    return try_call_except



############################
## OPTIMIZATION AND UTILS ##
############################


@log_function
def setup_optimizer(exp_params, model):
    """
    Initializing the optimizer object used to update the model parameters

    Args:
    -----
    exp_params: dictionary
        parameters corresponding to the different experiment
    model: nn.Module
        instanciated neural network model

    Returns:
    --------
    optimizer: Torch Optim object
        Initialized optimizer
    scheduler: Torch Optim object
        learning rate scheduler object used to decrease the lr after some epochs
    """

    lr = exp_params["training"]["lr"]
    scheduler = exp_params["training"]["scheduler"]
    scheduler_steps = exp_params["training"].get("scheduler_steps", 1e6)

    # Optimizer
    print_("Setting up Adam optimizer:")
    print_(f"    LR: {lr}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # LR-scheduler
    if(scheduler == "cosine_annealing"):
        print_("Setting up Cosine Annealing LR-Scheduler")
        print_(f"   Init LR: {lr}")
        print_(f"   T_max:   {scheduler_steps}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=scheduler_steps,
                eta_min=1e-7 #minimum learning rate
            )
    else:
        print_("Not using any LR-Scheduler")
        scheduler = IdentityScheduler(init_lr=lr)

    # seting up lr_warmup object
    lr_warmup = setup_lr_warmup(params=exp_params["training"])

    return optimizer, scheduler, lr_warmup


@log_function
def setup_lr_warmup(params):
    """
    Seting up the learning rate warmup handler given experiment params

    Args:
    -----
    params: dictionary
        training parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    lr_warmup: Object
        object that steadily increases the learning rate during the first iterations.
    """
    use_warmup = params["lr_warmup"]
    lr = params["lr"]
    if(use_warmup):
        warmup_steps = params["warmup_steps"]
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=warmup_steps)
        print_("Setting up learning rate warmup:")
        print_(f"  Target LR:     {lr}")
        print_(f"  Warmup Steps:  {warmup_steps}")
    else:
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=-1)
        print_("Not using learning rate warmup...")
    return lr_warmup


#
