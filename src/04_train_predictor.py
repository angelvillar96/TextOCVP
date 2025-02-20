"""
Training and Validation of an object-centric predictor module using a frozen
and pretrained object-centric video decomposition model
"""

import matplotlib
matplotlib.use('Agg')
import torch

from base.basePredictorTrainer import BasePredictorTrainer
from data.load_data import unwrap_batch_data
from lib.arguments import get_predictor_training_arguments
from lib.logger import Logger, print_
import lib.utils as utils
import lib.visualizations as vis

# hack to avoid weird port error in cluster
import multiprocessing
import multiprocessing.util
multiprocessing.util.abstract_sockets_supported = False
mgr = multiprocessing.Manager()



class Trainer(BasePredictorTrainer):
    """
   Training and Validation of an object-centric predictor module using a frozen
    and pretrained object-centric video decomposition model
    """

    def forward_loss_metric(self, batch_data, training=False, inference_only=False, **kwargs):
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
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        num_slots = self.model.module.num_slots
        slot_dim = self.model.module.slot_dim

        # fetching and checking data
        videos, others = unwrap_batch_data(self.exp_params, batch_data)
        videos = videos.to(self.device)
        B, L, C, H, W = videos.shape
        if L < num_context + num_preds:
            raise ValueError(f"Seq. length {L} must be >= {num_context + num_preds = }")
        videos = videos[:, :num_context + num_preds]

        # encoding frames into object slots usign pretrained object_centric model
        with torch.no_grad():
            out_model = self.model(
                    mode="decomp",
                    x=videos,
                    num_imgs=num_context+num_preds,
                    **others
                )
            slot_history = out_model.get("slot_history")
        # predicting future slots and rendering future frames
        pred_slots = self.predictor(slot_history, **others)
        pred_slots_dec = pred_slots.clone().reshape(B * num_preds, num_slots, slot_dim)
        out_decoder = self.model(
                mode="decode",
                slots=pred_slots_dec
            )
        pred_imgs = out_decoder.get("recons_imgs")
        pred_objs = out_decoder.get("recons", None)
        pred_masks = out_decoder.get("masks")
        pred_imgs = pred_imgs.view(B, num_preds, C, H, W)
        pred_masks = pred_masks.view(B, num_preds, num_slots, *pred_masks.shape[-3:])
        if pred_objs is not None:
            pred_objs = pred_objs.view(B, num_preds, num_slots, *pred_objs.shape[-3:])

        # Generating only model outputs
        out_model = (pred_imgs, pred_objs, pred_masks)
        if inference_only:
            return out_model, None

        # loss computation, backward pass and optimization
        target_slots = slot_history[:, num_context:num_context+num_preds, :, :]
        target_imgs = videos[:, num_context:num_context+num_preds, :, :]
        self.loss_tracker(
                pred_slots=pred_slots,
                target_slots=target_slots,
                pred_imgs=pred_imgs,
                target_imgs=target_imgs
            )
        loss = self.loss_tracker.get_last_losses(total_only=True)

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            if self.exp_params["training"]["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(
                        self.predictor.parameters(),
                        self.exp_params["training"]["clipping_max_value"]
                    )
            self.optimizer.step()

        return out_model, loss


    @torch.no_grad()
    def visualizations(self, batch_data, iter_):
        """
        Making visualizations with the current model to log on the tensorboard 
        """
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]

        # forward pass
        videos, others = unwrap_batch_data(self.exp_params, batch_data)
        out_model, _ = self.forward_loss_metric(
                batch_data=batch_data,
                training=False,
                inference_only=True
            )
        pred_imgs, pred_objs, pred_masks = out_model
        target_imgs = videos[:, num_context:num_context+num_preds, :, :]

        # visualizations
        for k in range(3):
            suptitle = None
            if "caption" in others:
                suptitle = others.get("caption")[k]
            fig, _ = vis.visualize_qualitative_eval(
                context=videos[k, :num_context],
                targets=target_imgs[k],
                preds=pred_imgs[k],
                suptitle=suptitle,
                savepath=None
            )
            self.writer.add_figure(tag=f"Qual Eval {k+1}", figure=fig, step=iter_+1)
            matplotlib.pyplot.close(fig)

            # visualize object reconstructions
            if pred_objs is not None:
                objs = pred_masks[k] * pred_objs[k]
            else:
                objs, _, _ = vis.process_objs_masks_dinosaur(
                    frames=pred_imgs,
                    masks=pred_masks,
                    H=96, W=96
                )[k]
            fig, _, _ = vis.visualize_decomp(
                    objs.clamp(0, 1),
                    savepath=None,
                    tag=f"Pred. Object Recons. {k+1}",
                    tb_writer=self.writer,
                    iter=iter_
                )
            matplotlib.pyplot.close(fig)
        return


if __name__ == "__main__":
    utils.clear_cmd()
    args = get_predictor_training_arguments()
    logger = Logger(exp_path=f"{args.exp_directory}/{args.name_pred_exp}")
    logger.log_info(
            message="Starting object-centric predictor training procedure",
            message_type="new_exp"
        )
    logger.log_arguments(args)

    print_("Initializing Predictor Trainer module")
    trainer = Trainer(
            exp_path=args.exp_directory,
            name_pred_exp=args.name_pred_exp,
            decomp_ckpt=args.decomp_ckpt,
            checkpoint=args.checkpoint,
            resume_training=args.resume_training
        )
    print_("Loading dataset...")
    trainer.load_data()
    print_("Setting up model, predictor and optimizer")
    trainer.load_decomp_model()
    trainer.setup_predictor()
    print_("Starting to train")
    trainer.training_loop()



