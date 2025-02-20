"""
Generating figures using a pretrained Object-Centric Decomposition model
"""

import os
import torch

from base.baseEvaluator import BaseEvaluator
from data.load_data import unwrap_batch_data
from lib.arguments import get_generate_figs_decomp_model_arguments
from lib.config import Config
from lib.logger import print_
import lib.utils as utils
import lib.visualizations as vis



class FigGenerator(BaseEvaluator):
    """
    Generating figures using a pretrained Object-Centric Decomposition model
    """
    
    def __init__(self, exp_path, decomp_ckpt, num_seqs=10):
        """
        Initializing the figure generation module
        """
        self.exp_path = os.path.join(exp_path)
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.decomp_ckpt = decomp_ckpt 
        self.num_seqs = num_seqs 
        self.model_name = self.exp_params["model"]["model_name"]
        self.batch_size = 1

        # direcoty where the figures will be saved
        model_name = decomp_ckpt.split('.')[0]
        self.plots_path = os.path.join(
                self.exp_path,
                "plots",
                f"FigGeneration_DecompModel={model_name}"
            )
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")        
        return


    @torch.no_grad()
    def compute_visualization(self, batch_data, img_idx):
        """
        Computing visualization
        """
        videos, _ = unwrap_batch_data(self.exp_params, batch_data)
        videos = videos.to(self.device)
        out_model = self.decomp_model(
                mode="decomp",
                x=videos,
                num_imgs=videos.shape[1]
            )
        recons_imgs = out_model.get("recons_imgs")
        recons_objs = out_model.get("recons_objs")
        recons_masks = out_model.get("masks")
    
        # directories for saving figures
        cur_dir = f"sequence_{img_idx:02d}"
        utils.create_directory(os.path.join(self.plots_path, cur_dir))

         # saving the reconstructed images
        N = min(10, videos.shape[1])
        savepath = os.path.join(self.plots_path, cur_dir, f"Recons_{img_idx+1}.png")
        vis.visualize_recons(
                imgs=videos[0, :N].clamp(0, 1),
                recons=recons_imgs[0, :N].clamp(0, 1),
                n_cols=10,
                savepath=savepath
            )

        # saving the reconstructed masks
        savepath = os.path.join(self.plots_path, cur_dir, f"masks_{img_idx+1}.png")
        _ = vis.visualize_decomp(
                recons_masks[0][:N],
                savepath=savepath,
                cmap="gray_r",
                vmin=0,
                vmax=1,
            )

        # saving the reconstructed objects
        savepath = os.path.join(self.plots_path, cur_dir, f"maskedObj_{img_idx+1}.png")
        if self.model_name == "SAVi":
            recon_combined = recons_masks[0, :N] * recons_objs[0, :N]
            recon_combined = torch.clamp(recon_combined, min=0, max=1)
        else:  # computing objects in DINO by multiplying the masks with the frames
            recon_combined, _, _ = vis.process_objs_masks_dinosaur(
                    frames=videos[:, :N],
                    masks=recons_masks[:, :N],
                    H=96, W=96
                )
            recon_combined = recon_combined[0]
        _ = vis.visualize_decomp(
                recon_combined,
                savepath=savepath,
                vmin=0,
                vmax=1,
            )
        return



if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_generate_figs_decomp_model_arguments()
    print_("Generating figures with an Object-Centric Decomposition model...")
    figGenerator = FigGenerator(
            exp_path=exp_path,
            decomp_ckpt=args.decomp_ckpt,
            num_seqs=args.num_seqs
        )
    print_("Loading dataset...")
    figGenerator.load_data()
    print_("Setting up model and loading pretrained parameters")
    figGenerator.load_decomp_model()
    print_("Generating and saving figures")
    figGenerator.generate_figs()


#
