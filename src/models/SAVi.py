"""
Implementation of the SAVi model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Blocks.attention import SlotAttention
from models.Blocks.initializers import get_initializer
from models.Blocks.model_blocks import SoftPositionEmbed
from models.Blocks.model_utils import init_xavier_
from models.Blocks.transition_models import get_transition_module
from models.EncodersDecoders.encoders import get_encoder
from models.EncodersDecoders.decoders import get_decoder



class SAVi(nn.Module):
    """
    SAVi model as described in the paper:
        - Kipf, et al. "Conditional object-centric learning from video." ICLR. 2022

    Args:
    -----
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
        Hidden and output dimension of the pointwise MLP encoder
    initializer: dict
        Parameters for the slot initializer
    encoder: dict
        Parameters for the image encoder
    decoder: dict
        Parameters for the slot decoder
    transition_module_params: dict
        Parameters for the transition moduke
    initializer: str
        Name of the slot initializer to use
    encoder: dict
        Parameters for the image encoder
    decoder: dict
        Parameters for the slot decoder
    transition_module_params: dict
        Parameters for the transition moduke
    """

    def __init__(self, num_slots, slot_dim, num_iterations=1, num_iterations_first=3,
                 in_channels=3, mlp_hidden=128, mlp_encoder_dim=128,
                 encoder={}, decoder={}, transition_module={}, initializer=None,
                 **kwargs):
        """ Model initializer """
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.in_channels = in_channels
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

        # encoder and decoder modules
        self.build_encoder(encoder_params=encoder)
        self.build_decoder(decoder_params=decoder)

        # slot attention
        self.slot_attention = SlotAttention(
                dim_feats=mlp_encoder_dim,
                dim_slots=slot_dim,
                num_slots=num_slots,
                num_iters_first=num_iterations_first,
                num_iters=num_iterations,
                mlp_hidden=mlp_hidden
            )
        self._init_model()
        return


    def build_encoder(self, encoder_params):
        """ Instanciating the image and MLP encoders """
        # Building convolutional encoder modules
        self.encoder = get_encoder(
                in_channels=self.in_channels,
                encoder=encoder_params
            )

        # MLP encoder applied prior to Slot Attention
        self.out_features = self.encoder.out_features
        self.encoder_pos_embedding = SoftPositionEmbed(
                hidden_size=self.out_features,
                resolution=encoder_params["encoder_params"].get("resolution")
            )
        self.encoder_mlp = nn.Sequential(
                nn.LayerNorm(self.out_features),
                nn.Linear(self.out_features, self.mlp_encoder_dim),
                nn.ReLU(),
                nn.Linear(self.mlp_encoder_dim, self.mlp_encoder_dim),
            )
        return


    def build_decoder(self, decoder_params):
        """ Instanciating decoder """        
        # decoder positional embedding and convolutional slot decoder
        self.decoder_resolution = decoder_params["decoder_params"].get("resolution")
        self.decoder_pos_embedding = SoftPositionEmbed(
                hidden_size=self.slot_dim,
                resolution=self.decoder_resolution
            )
        
        self.decoder = get_decoder(
                in_channels=self.slot_dim,
                decoder=decoder_params
            )
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
            Images to process with SAVi. Shape is (B, NumImgs, C, H, W)
        num_imgs: int
            Number of images to recursively encode into object slots.
        decode: bool
            If False, object slots are not decoded.
            This is useful during training of predictor modules, 
            when we only care about the object stats.

        Returns:
        --------
        recons_imgs: torch Tensor
            Rendered video frames by decoding and combining the slots.
            Shape is (B, num_imgs, C, H, W)
        recons_objs: torch Tensor
            Rendered objects by decoding slots.
            Shape is (B, num_imgs, num_slots, C, H, W)
        masks: torch Tensor
            Rendered object masks by decoding slots.
            Shape is (B, num_imgs, num_slots, 1, H, W)
        slot_history: torch Tensor
            Object slots encoded at every time step.
            Shape is (B, num_imgs, num_slots, slot_dim)
        """
        B = x.shape[0]
        slot_history, recons_history, ind_recons_history, masks_history = [], [], [], []

        # initializing slots
        predicted_slots = self.initializer(batch_size=B, **kwargs)

        # recursively mapping video frames into object slots
        for t in range(num_imgs):
            imgs = x[:, t]
            img_feats = self.encode(imgs)
            slots = self.slot_attention(
                    inputs=img_feats,
                    slots=predicted_slots,
                    step=t
                )  # slots ~ (B, N_slots, Slot_dim)
            # transition module
            predicted_slots = self.transition_module(slots)

            # decoding back to frames
            if decode:
                out_decoder = self.decode(slots)
                recon_combined = out_decoder.get("recons_imgs")
                recons = out_decoder.get("recons")
                masks = out_decoder.get("masks")
            else:
                recon_combined = torch.tensor([])
                recons = torch.tensor([])
                masks = torch.tensor([])
            
            # saving current reps
            slot_history.append(slots)
            recons_history.append(recon_combined)
            ind_recons_history.append(recons)
            masks_history.append(masks)

        model_out = {
            "recons_imgs": torch.stack(recons_history, dim=1),
            "recons_objs": torch.stack(ind_recons_history, dim=1),
            "masks": torch.stack(masks_history, dim=1),
            "slot_history": torch.stack(slot_history, dim=1),
        }
        return model_out


    def encode(self, x):
        """
        Encoding an image into image features
        """
        # encoding input frame and adding positional encodding
        x = self.encoder(x)  # x ~ (B,C,H,W)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos_embedding(x)  # x ~ (B,H,W,C)

        # further encodding with 1x1 Conv (implemented as shared MLP)
        x = torch.flatten(x, 1, 2)
        x = self.encoder_mlp(x)  # x ~ (B, N, Dim)
        return x

   
    def decode(self, slots):
        """
        Decoding slots into objects and masks
        """
        B = slots.shape[0]
        # adding broadcasing for the dissentangled decoder
        slots = self.broadcast(slots)
        y = self.decoder(slots)  # slots ~ (B*N_slots, Slot_dim, H, W)

        # recons and masks have shapes [B, N_S, C, H, W] & [B, N_S, 1, H, W] respectively
        y_reshaped = y.reshape(B, -1, self.in_channels + 1, y.shape[2], y.shape[3])
        recons, masks = y_reshaped.split([self.in_channels, 1], dim=2)

        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        out_decoder = {
            "recons_imgs": recon_combined,
            "recons": recons,
            "masks": masks,
        }
        return out_decoder


    def broadcast(self, slots):
        """ Broadcasting slots prior to decoding """
        slot_dim = slots.shape[-1]
        slots = slots.reshape((-1, 1, 1, slot_dim))
        slots = slots.repeat(
                (1, self.decoder_resolution[0], self.decoder_resolution[1], 1)
            )  # slots ~ (B * N_slots, H, W, Slot_dim)

        # adding positional embeddings to reshaped features
        slots = self.decoder_pos_embedding(slots)  # slots ~ (B * N_slots, H, W, Slot_dim)
        slots = slots.permute(0, 3, 1, 2)  # slots ~ (B * N_slots, Slot_dim, H, W)
        return slots


    @torch.no_grad()
    def _init_model(self):
        """
        Initalization of the model parameters
        Adapted from:
            https://github.com/addtt/object-centric-library/blob/main/models/slot_attention/trainer.py
        """
        init_xavier_(self)
        torch.nn.init.zeros_(self.slot_attention.gru.bias_ih)
        torch.nn.init.zeros_(self.slot_attention.gru.bias_hh)
        torch.nn.init.orthogonal_(self.slot_attention.gru.weight_hh)
        if hasattr(self.slot_attention, "slots_mu"):
            limit = math.sqrt(6.0 / (1 + self.slot_attention.dim_slots))
            torch.nn.init.uniform_(self.slot_attention.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slot_attention.slots_sigma, -limit, limit)
        return


#
