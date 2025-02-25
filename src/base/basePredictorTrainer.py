"""
Base predictor trainer from which all predictor trainer classes inherit.

Basically it removes the scaffolding that is repeat across all predictor
training modules
"""

import os
from tqdm import tqdm
import torch

from lib.config import Config
from lib.logger import print_, log_function, for_all_methods, log_info
from lib.loss import LossTracker
from lib.schedulers import WarmupVSScehdule
from lib.setup_model import emergency_save
import lib.setup_model as setup_model
import lib.utils as utils
import data as datalib
from models.Blocks.model_utils import freeze_params



@for_all_methods(log_function)
class BasePredictorTrainer:
    """
    Base Class for training and validating a predictor model

    Args:
    -----
    exp_path: string
        Path to the experiment directory from which to read the 
        experiment parameters, and where to store logs, plots and checkpoints
    name_pred_exp: string
        Name of the predictor experiment (subdirectory in parent directory) to train.
    decomp_ckpt: string
        Name of the pretrained object-centric video decomposition model used to
        extract object representation from frames and to decode the predicted
        slots back to images
    checkpoint: string/None
        Name of a model checkpoint stored in the models/ directory of the predictor
        experiment directory.
    resume_training: bool
        If True, saved checkpoint states from the optimizer, scheduler, ...
        are restored in order to continue training from the checkpoint
    """

    def __init__(self, name_pred_exp, exp_path, decomp_ckpt,
                 checkpoint=None, resume_training=False):
        """
        Initializing the trainer object
        """
        self.name_pred_exp = name_pred_exp
        self.parent_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_pred_exp)
        if not os.path.exists(self.exp_path):
            raise FileNotFoundError(f"Predictor {self.exp_path = } does not exist...")
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.decomp_ckpt = decomp_ckpt
        self.checkpoint = checkpoint
        self.resume_training = resume_training

        self.plots_path = os.path.join(self.exp_path, "plots", "valid_plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)

        self.training_losses = []
        self.validation_losses = []
        tboard_logs = os.path.join(
                self.exp_path,
                "tboard_logs",
                f"tboard_{utils.timestamp()}"
            )
        utils.create_directory(tboard_logs)
        self.writer = utils.TensorboardWriter(logdir=tboard_logs)
        return

    def load_data(self):
        """
        Loading train and validation datasets and fitting data-loader for iterating in batches
        """
        batch_size = self.exp_params["training"]["batch_size"]
        shuffle_train = self.exp_params["dataset"]["shuffle_train"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]

        # overriding sequence length of dataset with num_context + num_preds
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        new_seq_len = num_context + num_preds
        print_(f"Replacing sequence length with required seq. length of {new_seq_len}")
        self.exp_params["dataset"]["num_frames"] = new_seq_len

        train_set = datalib.load_data(exp_params=self.exp_params, split="train")
        print_(f"Examples in training set: {len(train_set)}")
        valid_set = datalib.load_data(exp_params=self.exp_params, split="valid")
        print_(f"Examples in validation set: {len(valid_set)}")
        self.train_loader = datalib.build_data_loader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=shuffle_train
            )
        self.valid_loader = datalib.build_data_loader(
                dataset=valid_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        return

    def load_decomp_model(self):
        """
        Load pretraiened video decomposition model from checkpoint
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()
        self.device_ids = list(range(num_gpus))
        print_(f"Using {torch.cuda.device_count()} GPUs")
        print_(f"  --> device: {self.device_ids}")

        # seting up model
        model = setup_model.setup_model(model_params=self.exp_params["model"])
        utils.log_architecture(
                model,
                exp_path=self.exp_path,
                fname="architecture_decomp_model.txt"
            )

        # loading pretrained parameters and freezing SAVi modules
        ckpt_path = os.path.join(self.parent_exp_path, "models", self.decomp_ckpt)
        print_(f"  --> Loading Decomp-Model params from {self.decomp_ckpt = }...")
        model = setup_model.load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                only_model=True
            )

        freeze_params(model)
        self.model = torch.nn.DataParallel(
                model.eval(),
                device_ids=self.device_ids
            ).to(self.device)
        return

    def setup_predictor(self):
        """
        Initializing predictor, optimizer, loss function and other related objects
        """
        # instanciating predictor model
        predictor = setup_model.setup_predictor(exp_params=self.exp_params)
        utils.log_architecture(
                predictor.predictor,  # .predictor to remove pred_wrapper
                exp_path=self.exp_path,
                fname="architecture_predictor.txt"
            )

        # loading optimizer, scheduler and loss
        optimizer, scheduler, lr_warmup = setup_model.setup_optimizer(
                exp_params=self.exp_params,
                model=predictor
            )
        loss_tracker = LossTracker(loss_params=self.exp_params["predictor_loss"])
        epoch = 0

        # loading pretrained model and other objects
        if self.checkpoint is not None:
            print_(f"  --> Loading predictor params from {self.checkpoint = }...")
            loaded_objects = setup_model.load_checkpoint(
                    checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                    model=predictor,
                    only_model=not self.resume_training,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    lr_warmup=lr_warmup
                )
            if self.resume_training:
                predictor, optimizer, scheduler, lr_warmup, epoch = loaded_objects
                print_(f"  --> Resuming training from epoch {epoch}...")
            else:
                predictor = loaded_objects

        self.predictor = torch.nn.DataParallel(
                predictor.eval(),
                device_ids=self.device_ids
            ).to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = epoch
        self.loss_tracker = loss_tracker
        self.warmup_scheduler = WarmupVSScehdule(
                optimizer=self.optimizer,
                lr_warmup=lr_warmup,
                scheduler=scheduler
            )
        return

    @emergency_save
    def training_loop(self):
        """
        Repearting the process validation epoch - train epoch for the
        number of epoch specified in the exp_params file.
        """
        num_epochs = self.exp_params["training"]["num_epochs"]
        save_frequency = self.exp_params["training"]["save_frequency"]

        # iterating for the desired number of epochs
        epoch = self.epoch
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            log_info(message=f"Epoch {epoch}/{num_epochs}")
            self.predictor.eval()
            self.valid_epoch(epoch)
            self.predictor.train()
            self.train_epoch(epoch)

            # adding to tensorboard plot containing both losses
            self.writer.add_scalars(
                    plot_name='Total Loss',
                    val_names=["train_loss", "eval_loss"],
                    vals=[self.training_losses[-1], self.validation_losses[-1]],
                    step=epoch+1
                )

            # updating learning rate scheduler or lr-warmup
            self.warmup_scheduler(
                    iter=-1,
                    epoch=epoch,
                    exp_params=self.exp_params,
                    end_epoch=True,
                    control_metric=self.validation_losses[-1]
                )

            # saving predictor checkpoint if reached saving frequency
            self.wrapper_save_checkpoint(
                    epoch=epoch,
                    savename="checkpoint_last_saved.pth"
                )
            if(epoch % save_frequency == 0 and epoch != 0):
                print_("Saving model checkpoint")
                self.wrapper_save_checkpoint(epoch=epoch, savedir="models")

        print_("Finished training procedure")
        print_("Saving final checkpoint")
        self.wrapper_save_checkpoint(epoch=epoch, finished=True)
        return

    def wrapper_save_checkpoint(self, epoch=None, savedir="models",
                                savename=None, finished=False):
        """
        Wrapper for saving a models in a more convenient manner
        """
        setup_model.save_checkpoint(
                model=self.predictor.module,
                optimizer=self.optimizer,
                scheduler=self.warmup_scheduler.scheduler,
                lr_warmup=self.warmup_scheduler.lr_warmup,
                epoch=epoch,
                exp_path=self.exp_path,
                savedir=savedir,
                savename=savename,
                finished=finished
            )
        return

    def train_epoch(self, epoch):
        """
        Training epoch loop
        """
        self.loss_tracker.reset()
        max_train_iters = self.exp_params["training"].get(
                "train_iters_per_epoch", len(self.train_loader)
            )
        total_progress_bar = min(len(self.train_loader), max_train_iters)
        progress_bar = tqdm(enumerate(self.train_loader), total=total_progress_bar)

        for i, data in progress_bar:
            if i >= max_train_iters:
                break
            iter_ = total_progress_bar * epoch + i
            self.warmup_scheduler(
                    iter=iter_,
                    epoch=epoch,
                    exp_params=self.exp_params,
                    end_epoch=False
                )

            # forward pass
            _, loss = self.forward_loss_metric(
                    batch_data=data,
                    training=True
                )

            # logging values to tensorboard
            if(iter_ % self.exp_params["training"]["log_frequency"] == 0):
                self.writer.log_full_dictionary(
                        dict=self.loss_tracker.get_last_losses(),
                        step=iter_,
                        plot_name="Train Loss",
                        dir="Train Loss Iter",
                    )
                self.writer.add_scalar(
                        name="Learning Rate",
                        val=self.optimizer.param_groups[0]['lr'],
                        step=iter_
                    )

            # logging visualizations
            if(iter_ % self.exp_params["training"]["image_log_frequency"] == 0):
                batch_data = next(iter(self.valid_loader))
                self.visualizations(batch_data=batch_data, iter_=iter_)

            # update progress bar
            progress_bar.set_description(
                    f"Epoch {epoch+1} iter {iter_}: train loss {loss.item():.5f}. "
                )

        self.loss_tracker.aggregate()
        average_loss_vals = self.loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
                dict=average_loss_vals,
                step=epoch + 1,
                plot_name="Train Loss",
                dir="Train Loss",
            )
        self.training_losses.append(average_loss_vals["_total"].item())
        return


    @torch.no_grad()
    def valid_epoch(self, epoch):
        """
        Validation epoch
        """
        self.loss_tracker.reset()
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        for i, data in progress_bar:
            _ = self.forward_loss_metric(
                    batch_data=data,
                    training=False
                )
            loss = self.loss_tracker.get_last_losses(total_only=True)
            progress_bar.set_description(
                    f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}. "
                )

        self.loss_tracker.aggregate()
        average_loss_vals = self.loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
                dict=average_loss_vals,
                step=epoch + 1,
                plot_name="Valid Loss",
                dir="Valid Loss",
            )
        self.validation_losses.append(average_loss_vals["_total"].item())
        return


    def forward_loss_metric(self, batch_data, training=False,
                            inference_only=False, **kwargs):
        """
        Computing a forwad pass through the model, and (if necessary) the loss
        values and optimziation

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch,
            including images, captions, or metadata, among others.
        training: bool
            If True, model is in training mode
        inference_only: bool
            If True, only forward pass through the model is performed.
        """
        raise NotImplementedError(
                "Base PredictorTrainer does not implement 'forward_loss_metric'..."
            )


    def visualizations(self):
        """
        Making visualizatios to log on the tensorboard

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch,
            including images, captions, or metadata, among others.
        """
        raise NotImplementedError(
                "Base PredictorTrainer does not implement 'visualizations'..."
            )
