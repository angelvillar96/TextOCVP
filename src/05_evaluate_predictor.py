"""
Evaluating a object-centric video prediction model for video prediction
or image-to-video generation.

This script can be used to evaluate the following OCVP-Models:
  - Vanilla OCVP (same as SlotFormer)
  - OCVP-Seq
  - OCVP-Par
  - TextOCVP-Custom
  - TextOCVP-T5
"""

import os
import torch

from data import unwrap_batch_data
from lib.arguments import get_predictor_evaluation_arguments
from lib.config import Config
from lib.logger import Logger, print_
import lib.utils as utils
from base.baseEvaluator import BaseEvaluator



class Evaluator(BaseEvaluator):
    """
    Evaluating a object-centric video prediction model for video prediction
    or image-to-video generation.
    """

    def __init__(self, exp_path, name_pred_exp, decomp_ckpt, pred_ckpt,
                 num_seed=None, num_preds=None, batch_size=None, results_name=None,
                 **kwargs):
        """ Evaluator initalizer """
        self.parent_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_pred_exp)
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.decomp_ckpt = decomp_ckpt
        self.pred_ckpt = pred_ckpt
        self.batch_size = batch_size
        self.results_name = results_name
        
        # paths and utils
        self.plots_path = os.path.join(self.exp_path, "plots")
        self.decomp_models_path = os.path.join(self.parent_exp_path, "models")
        self.models_path = os.path.join(self.exp_path, "models")
        self.override_num_seed_and_preds(num_seed=num_seed, num_preds=num_preds)
        self.set_metric_tracker()
        return


    @torch.no_grad()
    def forward_eval(self, batch_data, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch,
            including images, captions, or metadata, among others.

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        """
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        num_slots = self.decomp_model.module.num_slots
        slot_dim = self.decomp_model.module.slot_dim

        # fetching and preparing data
        videos, others = unwrap_batch_data(self.exp_params, batch_data)
        videos = videos.to(self.device)
        B, L, C, H, W = videos.shape
        if L < num_context + num_preds:
            raise ValueError(f"Seq. length {L} smaller that {num_context = } + {num_preds = }")

        # encoding images into object-centric slots
        out_model = self.decomp_model(
                mode="decomp",
                x=videos,
                num_imgs=num_context+num_preds,
                decode=False,
                **others
            )
        slot_history = out_model["slot_history"]

        # predicting future slots and decoding them into images
        pred_slots = self.predictor(slot_history, **others)
        pred_slots_decode = pred_slots.reshape(B * num_preds, num_slots, slot_dim)
        out_decoder = self.decomp_model(mode="decode", slots=pred_slots_decode)
        pred_imgs = out_decoder.get("recons_imgs")
        pred_imgs = pred_imgs.view(B, num_preds, C, H, W).clamp(0, 1)

        # evaluation
        targets = videos[:, num_context:num_context+num_preds].clamp(0, 1)
        self.metric_tracker.accumulate(
                preds=pred_imgs,
                targets=targets
            )
        return



if __name__ == "__main__":
    utils.clear_cmd()
    args = get_predictor_evaluation_arguments()

    logger = Logger(exp_path=f"{args.exp_directory}/{args.name_pred_exp}")
    logger.log_info(
            "Starting object-centric predictor evaluation procedure",
            message_type="new_exp"
        )
    logger.log_arguments(args)

    evaluator = Evaluator(
            exp_path=args.exp_directory,
            decomp_ckpt=args.decomp_ckpt,
            name_pred_exp=args.name_pred_exp,
            pred_ckpt=args.pred_ckpt,
            num_seed=args.num_seed,
            num_preds=args.num_preds,
            batch_size=args.batch_size,
            results_name=args.results_name,
        )
    print_("Loading dataset...")
    evaluator.load_data()
    print_("Setting up model and predictor and loading pretrained parameters")
    evaluator.load_decomp_model(models_path=evaluator.decomp_models_path)
    evaluator.load_predictor()
    # VIDEO PREDICTION EVALUATION
    print_("Starting video predictor evaluation")
    evaluator.evaluate()


#
