"""
Training and Validating a SAVi video decomposition model
"""

import matplotlib
matplotlib.use('Agg')
import torch

from base.baseTrainer import BaseTrainer
from data.load_data import unwrap_batch_data
from lib.arguments import get_train_decomp_arguments
from lib.logger import Logger, print_
import lib.utils as utils
from lib.visualizations import visualize_decomp, visualize_recons

# hack to avoid weird port error in cluster
import multiprocessing
import multiprocessing.util
multiprocessing.util.abstract_sockets_supported = False
mgr = multiprocessing.Manager()



class Trainer(BaseTrainer):
    """
    Class for training a SAVi model for object-centric video
    """

    def forward_loss_metric(self, batch_data, training=False, inference_only=False):
        """
        Computing a forwad pass through the model, and (if necessary) the loss
        values and optimization

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch,
            including images, captions, or metadata, among others.
        training: bool
            If True, model is in training mode
        inference_only: bool
            If True, only forward pass through the model is performed
        """
        videos, others = unwrap_batch_data(self.exp_params, batch_data)

        # forward pass
        videos = videos.to(self.device)
        out_model = self.model(
                x=videos,
                num_imgs=videos.shape[1],
                **others
            )

        if inference_only:
            return out_model, None

        # if necessary, computing loss, backward pass and optimization
        recons_imgs = out_model.get("recons_imgs")
        self.loss_tracker(
                pred_imgs=recons_imgs.clamp(0, 1),
                target_imgs=videos.clamp(0, 1)
            )

        loss = self.loss_tracker.get_last_losses(total_only=True)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            if self.exp_params["training"]["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.exp_params["training"]["clipping_max_value"]
                    )
            self.optimizer.step()

        return out_model, loss


    @torch.no_grad()
    def visualizations(self, batch_data, iter_):
        """
        Making a visualization using the current model and logging them to TBoard
        This function logs the following:
            - Ground-truth vs. Reconstructed Images
            - Objects and masks
        """
        if(iter_ % self.exp_params["training"]["image_log_frequency"] != 0):
            return

        videos, _ = unwrap_batch_data(self.exp_params, batch_data)
        out_model, _ = self.forward_loss_metric(
                batch_data=batch_data,
                training=False,
                inference_only=True
            )
        N = min(10, videos.shape[1])  # max of 10 frames for sleeker figures
        recons_history = out_model.get("recons_imgs")
        recons_objs = out_model.get("recons_objs")
        recons_masks = out_model.get("masks")

         # visualitations
        for k in range(3):
            # output reconstructions and input images
            visualize_recons(
                    imgs=videos[k][:N],
                    recons=recons_history[k][:N].clamp(0, 1),
                    savepath=None,
                    tb_writer=self.writer,
                    iter=iter_
                )

            # Rendered individual object masks
            fig, _, _ = visualize_decomp(
                    recons_masks[k][:N].clamp(0, 1),
                    savepath=None,
                    tag="masks",
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    tb_writer=self.writer,
                    iter=iter_
                )
            matplotlib.pyplot.close(fig)

            # Rendered individual combination of an object with its masks
            recon_combined = recons_masks[k][:N] * recons_objs[k][:N]
            fig, _, _ = visualize_decomp(
                    recon_combined.clamp(0, 1),
                    savepath=None,
                    tag="reconstruction_combined",
                    vmin=0,
                    vmax=1,
                    tb_writer=self.writer,
                    iter=iter_
                )
            matplotlib.pyplot.close(fig)

        return


if __name__ == "__main__":
    utils.clear_cmd()
    args = get_train_decomp_arguments()
    logger = Logger(exp_path=args.exp_directory)
    logger.log_info(
            message="Starting SAVi training procedure",
            message_type="new_exp"
        )
    logger.log_arguments(args)

    print_("Initializing SAVi Trainer...")
    trainer = Trainer(
            exp_path=args.exp_directory,
            checkpoint=args.checkpoint,
            resume_training=args.resume_training
        )
    print_("Setting up model and optimizer")
    trainer.setup_model()
    print_("Loading dataset...")
    trainer.load_data()
    print_("Starting to train")
    trainer.training_loop()


#
