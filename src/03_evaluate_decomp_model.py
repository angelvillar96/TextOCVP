"""
Evaluating an Object-Centric Video decomposition model checkpoint
for object-centric video decomposition and reconstruction
"""

import torch
from data import unwrap_batch_data
from lib.arguments import get_decomp_eval_arguments
from lib.logger import Logger, print_
import lib.utils as utils

from base.baseEvaluator import BaseEvaluator


class Evaluator(BaseEvaluator):
    """
    Evaluating an Object-Centric Video decomposition model checkpoint
    for object-centric video decomposition and reconstruction
    """

    @torch.no_grad()
    def forward_eval(self, batch_data, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch,
            including images, captions, or metadata, among others.
        """
        videos, others = unwrap_batch_data(self.exp_params, batch_data)
        videos = videos.to(self.device)
        out_model = self.decomp_model(
                x=videos,
                num_imgs=videos.shape[1],
                **others
            )
        recons_imgs = out_model.get("recons_imgs").clamp(0, 1)
        
        # evaluation
        self.metric_tracker.accumulate(
                preds=recons_imgs,
                targets=videos
            )
        return



if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_decomp_eval_arguments()
    logger = Logger(exp_path=exp_path)
    logger.log_info(
            message="Starting Decomposition Model evaluation procedure",
            message_type="new_exp"
        )

    print_("Initializing Evaluator...")
    logger.log_arguments(args)
    evaluator = Evaluator(
            exp_path=exp_path,
            decomp_ckpt=args.decomp_ckpt,
            results_name=args.results_name,
            batch_size=args.batch_size
        )
    print_("Loading dataset...")
    evaluator.load_data()
    print_("Setting up Decomposition Model and loading pretrained parameters")
    evaluator.load_decomp_model()
    print_("Starting visual quality evaluation")
    evaluator.evaluate()


#
