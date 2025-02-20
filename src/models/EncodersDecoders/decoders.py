"""
Factory of decoder modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.logger import print_
from models.Blocks.model_blocks import ConvBlock, Upsample


DECODERS = [
        "ConvDecoder",
        "MLPPatchDecoder"
    ]



def get_decoder(in_channels, decoder, **kwargs):
    """
    Instanciating a decoder given the model name and parameters
    """
    decoder_name = decoder["decoder_name"]
    decoder_params = decoder["decoder_params"]
    if decoder_name not in DECODERS:
        raise ValueError(f"Unknwon decoder_name {decoder_name}. Use one of {DECODERS}")

    if(decoder_name == "ConvDecoder"):
        decoder = ConvDecoder(
                in_channels=in_channels,
                hidden_dims=decoder_params.pop("num_channels"),
                kernel_size=decoder_params.pop("kernel_size"),
                upsample=decoder_params.pop("upsample"),
                out_channels=kwargs.get("out_channels", 4),
                **decoder_params
            )
    elif(decoder_name == "MLPPatchDecoder"):
        decoder = MLPPatchDecoder(**decoder_params)
    else:
        raise NotImplementedError(f"Unknown decoder {decoder_name}...")

    print_("Decoder:")
    print_(f"  --> Decoder={decoder_name}")
    print_(f"  --> in_channels={in_channels}")
    for k, v in kwargs.items():
        print_(f"  --> {k}={v}")
    return decoder



class ConvDecoder(nn.Module):
    """
    Simple fully convolutional decoder

    Args:
    -----
    in_channels: int
        Number of input channels to the decoder
    hidden_dims: list
        List with the hidden dimensions in the decoder. Final value is the number of output channels
    kernel_size: int
        Kernel size for the convolutional layers
    upsample: int or None
        If not None, feature maps are upsampled by this amount after every hidden convolutional layer
    """

    def __init__(self, in_channels, hidden_dims, kernel_size=5, upsample=None,
                 out_channels=4, **kwargs):
        """ Module initializer """
        super().__init__()
        self.in_channels = in_channels
        self.in_features = in_channels
        self.hidden_dims = hidden_dims
        self.out_features = hidden_dims[0]
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.upsample = None if upsample < 2 else upsample
        self.out_channels = out_channels

        self.decoder = self._build_decoder()
        return

    def _build_decoder(self):
        """
        Creating convolutional decoder given dimensionality parameters
        By default, it maps feature maps to a 5dim output, containing
        RGB objects and binary mask:
           (B,C,H,W)  -- > (B, N_S, 4, H, W)
        """
        modules = []
        in_channels = self.in_channels

        # adding convolutional layers to decoder
        for i in range(len(self.hidden_dims) - 1, -1, -1):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=self.hidden_dims[i],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                batch_norm=self.batch_norm,
            )
            in_channels = self.hidden_dims[i]
            modules.append(block)
            if isinstance(self.upsample, int) and self.upsample is not None and i > 0:
                modules.append(Upsample(scale_factor=self.upsample))
        # final conv layer
        final_conv = nn.Conv2d(
                in_channels=self.out_features,
                out_channels=self.out_channels,  # RGB + Mask
                kernel_size=3,
                stride=1,
                padding=1
            )
        modules.append(final_conv)

        decoder = nn.Sequential(*modules)
        return decoder

    def forward(self, x):
        """ Forward pass of the decoder """
        y = self.decoder(x)
        return y

    

class BasePatchDecoder(nn.Module):
    """ 
    Base decoder module that takes slot object representations in order to
    reconstruct ViT patch embeddings.
    
    Args:
    -----
    num_patches: int
        Number of patches extracted by the ViT encoder that we must predict
    in_dim: int
        Dimensionality of the object slot representations
    """
    
    def __init__(self, num_patches, in_dim):
        """ Initializer of the patch-decoder """
        super().__init__()
        self.num_patches = num_patches
        self.in_dim = in_dim
        self.pos_embed = nn.Parameter(
                torch.randn(1, 1, num_patches, in_dim) / (in_dim ** 0.5)
            )
        return

    def forward(self):
        """ Forward pass """
        raise NotImplementedError(f"Base Class 'PatchDecoder' does not implement forward pass")
    
    def broadcast_slots(self, slots):
        """
        Broadcasting (repeating) slots to match the number of patches N
        
        Args:
        -----
        slots: torch tensor
            Current state of object slots. Shape is (B, num_slots, slot_dim)
            
        Returns:
        --------
        broadcasted_slots: torch tensor
            Input slots broadcasted to the number of patches in the image.
            Shape is (B, num_slots, num_patches, slot_dim)
        """
        broadcasted_slots = slots.unsqueeze(2)
        broadcasted_slots = broadcasted_slots.repeat(1, 1, self.num_patches, 1)
        return broadcasted_slots
    
    def add_positional_encoding(self, slots):
        """
        Adding a learnable positional encoding to the broadcasted slots in order
        to augment them with positional information.
        
        Args:
        -----
        slots: torch tensor
            Broadcasted object slot representations.
            Shape is (B, num_slots, num_patches, slot_dim)

        Returns:
        -----
        augmented_slots: torch tensor
            Broadcasted object slot representations augmented with positional information.
            Shape is (B, num_slots, num_patches, slot_dim), i.e., same as in the input
        """
        B, num_slots, num_patches, slot_dim = slots.shape
        if num_patches != self.pos_embed.shape[2]:
            raise ValueError(f"{num_patches = } must be == {self.pos_embed.shape = }")
        if slot_dim != self.pos_embed.shape[3]:
            raise ValueError(f"{slot_dim = } must be == {self.pos_embed.shape = }")
        pos_embed = self.pos_embed.repeat(B, num_slots, 1, 1)
        augmented_slots = slots + pos_embed
        return augmented_slots
    

    
class MLPPatchDecoder(BasePatchDecoder):
    """
    MLP-based patch decoder used to predict ViT features from object slots.
    This module works similar to a Spatial Broadcast decoder:
      1- Slots are repeated for each spatial position
      2- A positional encoding is added to augmented the broadcasted slots
         with positional information
      3- Broadcased slots are decoded independently with an MLP to predict
         features and alpha masks
      4- Final reconstruction is achieved via a weighted sum across slots
    
    Args:
    -----
    num_patches: int
        Number of patches extracted by the ViT encoder, and that we must predict
    in_dim: int
        Dimensionality of the object slot representations, D_slots
    hidden_dim: int
        Hidden dimension of the MLP decoder
    out_dim: int
        Dimensionality of the ViT features to predict, D_feat
    num_layers: int
        Number of Linear layers in the MLP decoder
    initial_layer_norm: bool
        If True, a Layer Normalization is applied prior to the MLP
    reconstruct_images: bool
        If True, a small CNN decoder will be used to aslo reconstruct images
    """
    
    def __init__(self, num_patches, in_dim, hidden_dim, out_dim, num_layers=4,
                 initial_layer_norm=False, reconstruct_images=False, **kwargs):
        """ Module initializer """
        super().__init__(
                num_patches=num_patches,
                in_dim=in_dim
            )
        # MLP for reconstructing features
        self.num_patches = num_patches
        self.patch_grid = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim  # Num_Feats + alpha
        self.num_layers = num_layers  # for MLP decoder
        self.initial_layer_norm = initial_layer_norm
        self.mlp = self._build_mlp()
        
        # CNN for image reconstruction
        self.reconstruct_images = reconstruct_images
        if self.reconstruct_images:
            self.patch_size = kwargs.get("patch_size")
            self.image_size = kwargs.get("img_size")
            self.num_layers_cnn = kwargs.get("num_layers_cnn")
            # instanciating CNN
            self.conv_patch_decoder = self._build_conv_patch_decoder(
                    in_dim= out_dim - 1,  # remove alpha masks
                    hidden_dim= self.hidden_dim, # using the same as self.mlp
                    num_layers= self.num_layers_cnn,
                    patch_size= self.patch_size
                )
        return
    

    def forward(self, slots):
        """
        Forward pass through MLP-patch Decoder module
        
        Args:
        -----
        slots: torch Tensor
            Object slot representations used to decode the patch DINO features from.
            Shape is (B, num_slots, slot_dim)
        """
        B, num_slots = slots.shape[0], slots.shape[1]
        # broadcast slots and decoding with MLP
        broadcasted_slots = self.broadcast_slots(slots)
        augmented_slots = self.add_positional_encoding(broadcasted_slots)
        decoded_features = self.mlp(augmented_slots)
        
        # rendering patch features via weighted sum
        feats, alpha = decoded_features[..., :-1], decoded_features[..., -1:] 
        alpha = F.softmax(alpha, dim=1)
        recons_features = torch.sum(feats * alpha, dim=1)
        masks = alpha.reshape(B, num_slots, 1, *self.patch_grid)

        # Image rendering with CNN decoder
        recons_imgs = torch.tensor([])
        if self.reconstruct_images:
            # assembing patch grid of features 
            input_decoder = recons_features.permute(0, 2, 1)
            input_decoder = input_decoder.reshape(B, self.out_dim-1, *self.patch_grid)
            recons_imgs = self.conv_patch_decoder(input_decoder)
            # matching exact image size
            if recons_imgs.shape[-1] != self.image_size:
                recons_imgs = F.interpolate(
                        recons_imgs,
                        size=(self.image_size, self.image_size),
                        mode='bilinear',
                        align_corners=False
                    ).contiguous()

        out = {
                "recons_imgs": recons_imgs,
                "recons_feats": recons_features,
                "masks": masks,
            }
        return out


    def _build_mlp(self):
        """ Instanciating MLP decoder module """
        mlp = []
        if self.initial_layer_norm:
            mlp.append(nn.LayerNorm(self.in_dim))
        for i in range(self.num_layers):
            dim1 = self.hidden_dim if i > 0 else self.in_dim
            dim2 = self.hidden_dim if i < self.num_layers - 1 else self.out_dim
            mlp.append(nn.Linear(dim1, dim2))
            if i < self.num_layers - 1:
                mlp.append(nn.ReLU())
        mlp = nn.Sequential(*mlp)
        return mlp
    

    def _build_conv_patch_decoder(self, in_dim, hidden_dim, num_layers, patch_size):
        """
        Build a simple patch convolutional decoder to map reconstructed patch features
        to reconstructed patch pixel values.
        """
        modules = []
        current_size = self.patch_grid[0]  # initial spatial dim of the input
        
        # adding convolutional layers to decoder
        for i in range(num_layers):
            # determine input and output channels for the current layer
            in_channels = in_dim if i == 0 else hidden_dim
            if (i > 0) and ((i + 1) * 2 < patch_size) and (current_size < self.image_size):
                hidden_dim = hidden_dim // 2  # decrease num_channels after upsampling
            # conv block
            block = ConvBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    batch_norm=True,
                )
            modules.append(block)
            # upsampling
            if ((i + 1) * 2 < patch_size) and (current_size < self.image_size):
                modules.append(Upsample(scale_factor=2))
                current_size = current_size*2
        
        # final conv layer
        final_conv = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1
            )
        modules.append(final_conv)

        conv_decoder = nn.Sequential(*modules)
        return conv_decoder



