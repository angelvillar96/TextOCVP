"""
Generating some figures using a pretrained decomposition model and
an object-centric video prediction or image-to-video generation model
"""

import os

import matplotlib.pyplot as plt
import torch.nn.functional
from tqdm import tqdm
import torch

from base.baseEvaluator import BaseEvaluator
from data.load_data import unwrap_batch_data
from lib.arguments import get_generate_figs_pred
from lib.config import Config
from lib.logger import print_
from lib.metrics import MetricTracker
import lib.utils as utils
import lib.visualizations as vis



class FigGenerator(BaseEvaluator):
    """
    Generating some figures using a pretrained decomposition model and
    an object-centric video prediction or image-to-video generation model
    """

    def __init__(self, exp_path, name_pred_exp, decomp_ckpt, pred_ckpt,
                 num_seqs=30, num_preds=25):
        """
        Initializing the Figure Generation module
        """
        self.parent_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_pred_exp)
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.decomp_ckpt = decomp_ckpt
        self.pred_ckpt = pred_ckpt
        self.num_seqs = num_seqs
        self.override_num_seed_and_preds(num_seed=None, num_preds=num_preds)
        self.decomp_models_path = os.path.join(self.parent_exp_path, "models")
        self.models_path = os.path.join(self.exp_path, "models")
        self.batch_size = 1

        # creating directory where to store the figures
        self.pred_name = self.exp_params["predictor"]["predictor_name"]
        self.plots_dir = \
                f"FigGen_Pred_{self.pred_name}_{name_pred_exp.split('/')[-1]}_" + \
                f"{pred_ckpt[:-4]}_" + \
                f"NumPreds={num_preds}"
        self.plots_path = os.path.join(self.exp_path, "plots", self.plots_dir)
        utils.create_directory(self.plots_path)
        return


    @torch.no_grad()
    def generate_figs(self):
        """
        Evaluating model epoch loop
        """
        utils.set_random_seed()
        num_seed = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        metric_tracker = MetricTracker(exp_path=None, metrics=["psnr", "lpips"])

        test_loader_iterator = iter(self.test_loader)
        for idx in tqdm(range(self.num_seqs)):
            batch_data = next(test_loader_iterator)
            videos, others = unwrap_batch_data(self.exp_params, batch_data)
            videos = videos.to(self.device)

            n_frames = videos.shape[1]
            if n_frames < num_seed + num_preds:
                raise ValueError(f"{n_frames = } must be >= {num_seed + num_preds = }")
            videos = videos[:, :num_seed + num_preds]

            # forward pass through object-centric prediction model
            out_model = self.forward_pass(videos, **others)
            pred_imgs, pred_objs, pred_masks, seed_objs, seed_masks = out_model

            # computing metrics for sequence to visualize
            metric_tracker.reset_results()
            metric_tracker.accumulate(
                    preds=pred_imgs.clamp(0, 1),
                    targets=videos[:1, num_seed:num_seed+num_preds].clamp(0, 1)
                )
            metric_tracker.aggregate()
            results = metric_tracker.get_results()
            psnr, lpips = results["psnr"]["mean"], results["lpips"]["mean"]
            cur_dir = f"img_{idx+1}_psnr={round(psnr,2)}_lpips={round(lpips, 3)}"

            # generating and saving visualizations
            self.compute_visualization(
                    videos=videos,
                    seed_objs=seed_objs,
                    seed_masks=seed_masks,
                    pred_imgs=pred_imgs,
                    pred_objs=pred_objs,
                    pred_masks=pred_masks,
                    cur_dir=cur_dir,
                    **others
                )
        return


    @torch.no_grad()
    def forward_pass(self, videos, **others):
        """
        Forward pass through SAVi and Preditor
        """
        B, L, C, H, W = videos.shape
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        num_slots = self.decomp_model.module.num_slots
        slot_dim = self.decomp_model.module.slot_dim

        out_model = self.decomp_model(
                mode="decomp",
                x=videos,
                num_imgs=L,
                decode=True
            )
        slot_history = out_model["slot_history"]
        seed_objs = out_model.get("recons_objs", None)
        seed_masks = out_model["masks"]        

        # predicting future slots and decoding predicted images 
        pred_slots = self.predictor(slot_history, **others)
        pred_slots_decode = pred_slots.reshape(B * num_preds, num_slots, slot_dim)
        decoder_out = self.decomp_model(
                mode="decode",
                slots=pred_slots_decode
            )
        pred_imgs = decoder_out.get("recons_imgs")
        pred_masks = decoder_out.get("masks")
        pred_objs = decoder_out.get("recons_objs", None)
        
        pred_imgs = pred_imgs.view(B, num_preds, C, H, W).clamp(0, 1)
        pred_masks = pred_masks.view(B, num_preds, num_slots, *pred_masks.shape[-3:])
        if pred_objs is not None:
            pred_objs = pred_objs.view(B, num_preds, num_slots, *pred_objs.shape[-3:])

        return pred_imgs, pred_objs, pred_masks, seed_objs, seed_masks


    def compute_visualization(self, videos, seed_objs, seed_masks, pred_imgs,
                              pred_objs, pred_masks, cur_dir, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics
        """
        utils.create_directory(self.plots_path, cur_dir)

        # some hpyer-parameters of the video model
        num_seed = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
    
        # in extended-DINOSAUR, we have no object images, so we create them
        # additionally, we resize stuff to (96,96) for faster processing
        if seed_objs is None or pred_objs is None:
            seed_objs, seed_masks, videos_tiny = vis.process_objs_masks_dinosaur(
                    frames=videos[:, :num_seed],
                    masks=seed_masks[:, :num_seed],
                    H=96, W=96
                )
            pred_objs, pred_masks, pred_imgs_tiny = vis.process_objs_masks_dinosaur(
                    frames=pred_imgs,
                    masks=pred_masks,
                    H=96, W=96
                )
            videos_overlay = torch.cat([videos_tiny, pred_imgs_tiny], dim=1)
        else:
            videos_overlay = videos
            
        # processing objs
        seed_imgs = videos[:, :num_seed, :, :]
        seed_objs = seed_objs[:, :num_seed]
        seed_masks = seed_masks[:, :num_seed]
        target_imgs = videos[:, num_seed:num_seed+num_preds, :, :]

        # aligned objects (seed and pred)
        seed_objs = vis.add_border(seed_objs * seed_masks, color_name="green", pad=2)
        pred_objs = vis.add_border(pred_objs * pred_masks, color_name="red", pad=2)
        all_objs = torch.cat([seed_objs, pred_objs], dim=1)
        fig, _ = vis.visualize_aligned_slots(
                all_objs[0],
                savepath=os.path.join(self.plots_path, cur_dir, "aligned_slots.png")
            )
        plt.close(fig)

        # Video predictions
        fig, _ = vis.visualize_qualitative_eval(
                context=seed_imgs[0],
                targets=target_imgs[0],
                preds=pred_imgs[0],
                savepath=os.path.join(self.plots_path, cur_dir, "qual_eval_rgb.png")
            )
        plt.close(fig)

        # processing masks into segmentations
        seed_masks_categorical = seed_masks[0].argmax(dim=1)
        pred_masks_categorical = pred_masks[0].argmax(dim=1)
        all_masks_categorical = torch.cat(
                [seed_masks_categorical, pred_masks_categorical],
                dim=0
            )
        masks_vis = vis.masks_to_rgb(x=all_masks_categorical)[:, 0]

        # overlaying masks on images
        masks_categorical_channels = vis.idx_to_one_hot(x=all_masks_categorical[:, 0])
        disp_overlay = vis.overlay_segmentations(
                videos_overlay[0].cpu().detach(),
                masks_categorical_channels.cpu().detach(),
                colors=vis.COLORS,
                alpha=0.6
            )

        # Sequence GIFs
        gt_frames = torch.cat([seed_imgs, target_imgs], dim=1)
        pred_frames = torch.cat([seed_imgs, pred_imgs], dim=1)
        vis.make_gif(
                gt_frames[0],
                savepath=os.path.join(self.plots_path, cur_dir, "gt_GIF_frames.gif"),
                n_seed=1000,
                use_border=True
            )
        vis.make_gif(
                pred_frames[0],
                savepath=os.path.join(self.plots_path, cur_dir, "pred_GIF_frames.gif"),
                n_seed=num_seed,
                use_border=True
            )
        vis.make_gif(
                masks_vis,
                savepath=os.path.join(self.plots_path, cur_dir, "masks_GIF_masks.gif"),
                n_seed=num_seed,
                use_border=True
            )
        vis.make_gif(
                disp_overlay,
                savepath=os.path.join(self.plots_path, cur_dir, "overlay_GIF.gif"),
                n_seed=num_seed,
                use_border=True
            )

        # Object GIFs
        for obj_id in range(all_objs.shape[2]):
            vis.make_gif(
                    all_objs[0, :, obj_id],
                    savepath=os.path.join(self.plots_path, cur_dir, f"gt_obj_{obj_id+1}.gif"),
                    n_seed=num_seed,
                    use_border=False
                )
        
        # saving text prompt
        if "caption" in kwargs:
            caption = kwargs.get("caption")
            with open(os.path.join(self.plots_path, cur_dir, "prompt.txt"), 'w') as f:
                f.write(caption[0])

        return


if __name__ == "__main__":
    utils.clear_cmd()
    args = get_generate_figs_pred()
    print_("Generating figures for predictor model...")
    figGenerator = FigGenerator(
            exp_path=args.exp_directory,
            decomp_ckpt=args.decomp_ckpt,
            name_pred_exp=args.name_pred_exp,
            pred_ckpt=args.pred_ckpt,
            num_seqs=args.num_seqs,
            num_preds=args.num_preds,
        )
    print_("Loading dataset...")
    figGenerator.load_data()
    print_("Setting up model and loading pretrained parameters")
    figGenerator.load_decomp_model(models_path=figGenerator.decomp_models_path)
    print_("Setting up predictor and loading pretrained parameters")
    figGenerator.load_predictor()
    print_("Generating and saving figures")
    figGenerator.generate_figs()


#
