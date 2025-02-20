"""
Implementation of Extended-DINOSAUR object-centric video decomposition model
"""

import math
import torch
import torch.nn as nn

from models.Blocks.attention import SlotAttention
from models.Blocks.initializers import get_initializer
from models.Blocks.model_utils import init_xavier_, freeze_params
from models.Blocks.transition_models import get_transition_module
from models.EncodersDecoders.encoders import get_encoder
from models.EncodersDecoders.decoders import get_decoder



class ExtendedDINOSAUR(nn.Module):
    """
    Implementation of Extended-DINOSAUR object-centric video decomposition model.
    This model is trained to jointly reconstruct features from a DINO-pretraiend
    backbone, as well as the input video frames.

    Args:
    -----
    img_size: integer
        Size of the input images (assumed to be square)
    num_slots: integer
        number of object slots to use. Corresponds to N-objects + background
    slot_dim: integer
        Dimensionality of the object slot embeddings
    num_iterations: integer
        number of recurrent iterations in Slot Attention for slot refinement in 
        all but the first frame in the sequence
    num_iterations_first: none/interger
        Number of recurrent iterations for the first frame in the sequence.
    in_channels: integer
        number of input (e.g., RGB) channels
    mlp_hidden: int
        Hidden dimension of the MLP in the slot attention module
    mlp_encoder_dim: int
        Dimensionality of the encoded patch tokens.
    initializer: str
        Name of the slot initializer to use
    encoder: dict
        Parameters for the image encoder
    decoder: dict
        Parameters for the slot decoder
    transition_module_params: dict
        Parameters for the transition moduke
    """

    def __init__(self, img_size, num_slots, slot_dim, num_iterations=1,
                 num_iterations_first=3, in_channels=3, mlp_hidden=128,
                 mlp_encoder_dim=128, initializer=None,
                 encoder=None, decoder=None, transition_module=None, **kwargs):
        """ Model initializer """
        super().__init__()
        self.img_size = img_size
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations_first = num_iterations_first
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.mlp_hidden = mlp_hidden
        self.mlp_encoder_dim = mlp_encoder_dim    

        # building slot initializer
        self.initializer = get_initializer(
                mode=initializer,
                slot_dim=slot_dim,
                num_slots=num_slots
            )

        # transition module, e.g., transformer
        self.transition_module = get_transition_module(
                slot_dim=slot_dim,
                **transition_module
            )
        
        # Building encoder modules
        if self.img_size is None:
            raise KeyError(
                    "'img_size' must be provided in model parameters in order to " + 
                    "instanciate ViT-based image encoder."
                )
        if "vit" not in encoder["encoder_name"]:
            raise NameError(f"Extended-DINOSAUR expects a ViT-Based encoder...")
        encoder["encoder_params"]["img_size"]= self.img_size
        self.encoder = get_encoder(
                in_channels=in_channels,
                encoder=encoder,
            )
        freeze_params(self.encoder)
        
        # MLP that projects ViT patch features before Slot Attention
        self.linear_feat_proj = nn.Sequential(
                nn.LayerNorm(mlp_encoder_dim),
                nn.Linear(mlp_encoder_dim, mlp_encoder_dim),
                nn.ReLU(),
                nn.Linear(mlp_encoder_dim, slot_dim),
            )

        # decoder module
        if decoder["decoder_name"] != "MLPPatchDecoder":
            raise NameError(f"Extended-DINOSAUR expects a 'MLPPatchDecoder'...")
        decoder["decoder_params"]["img_size"]= self.img_size
        self.decoder = get_decoder(
                in_channels=in_channels,
                decoder=decoder
            )

        # slot attention corrector
        self.slot_attention = SlotAttention(
                dim_feats=slot_dim,
                dim_slots=slot_dim,
                num_slots=num_slots,
                num_iters_first=num_iterations_first,
                num_iters=num_iterations,
                mlp_hidden=mlp_hidden,
            )
        self._init_model()
        return


    def forward(self, mode="decomp", *args, **kwargs):
        """
        Forward class that orquestrates calling the decomposition or decoding-only.
        This is needed for Data-Parallelization
        """
        if mode == "decomp":
           return self.forward_decomp(*args, **kwargs)
        elif mode == "decode":
           return self.decode(*args, **kwargs)
        else:
            raise NameError(f"{mode = } not recognized. Use ['decomp', 'decode']")


    def forward_decomp(self, x, num_imgs=10, decode=True, **kwargs):
        """
        Forward pass through the model

        Args:
        -----
        x: torch Tensor
            Images to process with Extended-DINOSAUR.
            Shape is (B, NumImgs, C, H, W)
        num_imgs: int
            Number of images to recursively encode into object slots.
        decode: bool
            If False, object slots are not decoded.
            This is useful during training of predictor modules, 
            when we only care about the object stats.

        Returns:
        --------
        recons_imgs: torch Tensor
            Rendered video frames by decoding slots and DINO feats.
            Shape is (B, num_imgs, C, H, W)
        recons_feats: torch Tensor
            Rendered DINO feats obtained by decoding the slots.
            Shape is (B, num_frames, patch_dim, NumPatches_H, NumPatches_W)
        slot_history: torch Tensor
            Object slots encoded at every time step.
            Shape is (B, num_imgs, num_slots, slot_dim)
        encoded_img_feats: torch Tensor
            Patch features encoded with DINOv2.
            This is one of the targets for training Extended-DINOSAUR.
            Shape is (B, num_frames, patch_dim, NumPatches_H, NumPatches_W)
        recons_obj_feats: torch Tensor
            Per-object reconstructed features.
            Shape is (B, num_frames, num_objs, patch_dim, NumPatches_H, NumPatches_W)
        masks: torch Tensor
            Per-object reconstructed alpha masks.
            Shape is (B, num_frames, num_objs, 1, NumPatches_H, NumPatches_W)
        """
        # initializing slots
        B = x.shape[0]
        predicted_slots = self.initializer(batch_size=B, **kwargs)
        outs = []

        # recursively mapping video frames into object slots
        for t in range(num_imgs):
            cur_outs = {}
            imgs = x[:, t]
            # encoding with DINO-ViT and projection to slot dim
            with torch.no_grad():
                img_feats = self.encoder(imgs)
            proj_img_feats = self.linear_feat_proj(img_feats)
            cur_outs["encoded_img_feats"] = img_feats

            # slot attention and transition module
            slots = self.slot_attention(
                    inputs=proj_img_feats,
                    slots=predicted_slots,
                    step=t
                )  # slots ~ (B, N_slots, Slot_dim)
            cur_outs["slot_history"] = slots
            predicted_slots = self.transition_module(slots)
            
            # decoding slots if necessary
            if decode:
                out_decoder = self.decode(slots)
                cur_outs = {**cur_outs, **out_decoder}
            outs.append(cur_outs)

        outs = {k: torch.stack([d[k] for d in outs], dim=1) for k in outs[0].keys()}
        return outs


    def decode(self, slots):
        """ Wrapper for decoding slots into patch features, and pottentially images """
        out_decoder = self.decoder(slots)
        return out_decoder


    @torch.no_grad()
    def _init_model(self):
        """
        Initalization of the model parameters

        Adapted from:
            https://github.com/addtt/object-centric-library/blob/main/models/slot_attention/trainer.py
        """
        init_xavier_(self.linear_feat_proj)
        init_xavier_(self.transition_module)
        init_xavier_(self.slot_attention)
        init_xavier_(self.decoder)

        torch.nn.init.zeros_(self.slot_attention.gru.bias_ih)
        torch.nn.init.zeros_(self.slot_attention.gru.bias_hh)
        torch.nn.init.orthogonal_(self.slot_attention.gru.weight_hh)
        if hasattr(self.slot_attention, "slots_mu"):
            limit = math.sqrt(6.0 / (1 + self.slot_attention.dim_slots))
            torch.nn.init.uniform_(self.slot_attention.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slot_attention.slots_sigma, -limit, limit)
        return



#
