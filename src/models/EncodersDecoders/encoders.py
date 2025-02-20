"""
Factory of encoder modules
"""


import torch.nn as nn

from lib.logger import print_
from models.Blocks.model_blocks import ConvBlock
import models.EncodersDecoders.timm_encoders as timm_encoders 


ENCODERS = [
        "ConvEncoder",
        "ConvEncoder128",
        "ResNet",
        "vit_small_patch16_224_dino",
        "vit_small_patch8_224_dino",
        "vit_base_patch16_224_dino",
        "vit_base_patch8_224_dino",
        "vit_small_patch14_dinov2",
        "vit_base_patch14_dinov2"
    ]



def get_encoder(in_channels, encoder, **kwargs):
    """
    Instanciating an encoder given the model name and parameters
    """
    encoder_name = encoder["encoder_name"]
    encoder_params = encoder["encoder_params"]
    if encoder_name not in ENCODERS:
        raise ValueError(f"Unknwon {encoder_name = }. Use one of {ENCODERS}")
    

    if encoder_name == "ConvEncoder":
        encoder = SimpleConvEncoder(
                in_channels=in_channels,
                hidden_dims=encoder_params.pop("num_channels"),
                kernel_size=encoder_params.pop("kernel_size"),
            )
    elif encoder_name == "ConvEncoder128":
        encoder = ConvEncoder128()
    elif encoder_name == "vit_small_patch16_224_dino":
        img_size = encoder_params.get("img_size")
        vit_backbone = timm_encoders.vit_small_patch16_224_dino(img_size=img_size)
        encoder = timm_encoders.ViTEncoder(
                vit_backbone=vit_backbone,
                num_blocks=encoder_params.get("num_blocks")
            )
    elif encoder_name == "vit_small_patch8_224_dino":
        img_size = encoder_params.get("img_size")
        vit_backbone = timm_encoders.vit_small_patch8_224_dino(img_size=img_size)
        encoder = timm_encoders.ViTEncoder(
                vit_backbone=vit_backbone,
                num_blocks=encoder_params.get("num_blocks")
            )
    elif encoder_name == "vit_base_patch16_224_dino":
        img_size = encoder_params.get("img_size")
        vit_backbone = timm_encoders.vit_base_patch16_224_dino(img_size=img_size)
        encoder = timm_encoders.ViTEncoder(
                vit_backbone=vit_backbone,
                num_blocks=encoder_params.get("num_blocks")
            )
    elif encoder_name == "vit_base_patch8_224_dino":
        img_size = encoder_params.get("img_size")
        vit_backbone = timm_encoders.vit_base_patch8_224_dino(img_size=img_size)
        encoder = timm_encoders.ViTEncoder(
                vit_backbone=vit_backbone,
                num_blocks=encoder_params.get("num_blocks")
            )
    elif encoder_name == "vit_small_patch14_dinov2":
        img_size = encoder_params.get("img_size")
        vit_backbone = timm_encoders.vit_small_patch14_dinov2(img_size=img_size)
        encoder = timm_encoders.ViTEncoder(
                vit_backbone=vit_backbone,
                num_blocks=encoder_params.get("num_blocks")
            )
    elif encoder_name == "vit_base_patch14_dinov2":
        img_size = encoder_params.get("img_size")
        vit_backbone = timm_encoders.vit_base_patch14_dinov2(img_size=img_size)
        encoder = timm_encoders.ViTEncoder(
                vit_backbone=vit_backbone,
                num_blocks=encoder_params.get("num_blocks")
            )
    else:
        raise NotImplementedError(f"Unknown encoder {encoder_name}...")

    print_("Encoder:")
    print_(f"  --> Encoder_type={encoder_name}")
    print_(f"  --> in_channels={in_channels}")
    for k, v in kwargs.items():
        print_(f"  --> {k}={v}")
    return encoder



class SimpleConvEncoder(nn.Module):
    """
    Simple fully convolutional encoder

    Args:
    -----
    in_channels: integer
        number of input (e.g., RGB) channels
    kernel_size: integer
        side of the CNN filters
    hidden_dims: list/tuple of integers (int, ..., int)
        number of output channels for each conv layer
    """

    def __init__(self, in_channels=3, hidden_dims=(64, 64, 64, 64), kernel_size=5, **kwargs):
        """
        Module initializer
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.use_downsample = kwargs.get("downsample_encoder", False)
        self.downsample = kwargs.get("downsample", 2)

        self.out_features = hidden_dims[-1]
        self.encoder = self._build_encoder()
        return

    def _build_encoder(self):
        """
        Creating convolutional encoder given dimensionality parameters
        """
        modules = []
        in_channels = self.in_channels
        for i, h_dim in enumerate(self.hidden_dims):
            if self.use_downsample and i < len(self.hidden_dims) - 1:
                downsample = self.downsample
            else:
                downsample = False
            block = ConvBlock(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.kernel_size // 2,
                    batch_norm=self.batch_norm,
                    max_pool=downsample,
                    activation=True
                )
            modules.append(block)
            in_channels = h_dim
        encoder = nn.Sequential(*modules)
        return encoder

    def forward(self, x):
        """ Forward pass """
        y = self.encoder(x)
        return y
    
    

class ConvEncoder128(nn.Module):
    """ Simple fully convolutional encoder for 128x128 images """

    def __init__(self):
        """ Module initializer """
        super().__init__()
        self.out_features = 64
        self.encoder = self._build_encoder()
        return

    def _build_encoder(self):
        """
        Creating convolutional encoder given dimensionality parameters
        """
        modules = []
        for i in range(4):
            use_activation = True if i < 3 else False
            in_channels = 3 if i == 0 else 64
            out_channels = 64
            block = ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    stride=2 if i == 0 else 1,
                    padding=2,
                    batch_norm=False,
                    max_pool=None,
                    activation=use_activation
                )
            modules.append(block)
        encoder = nn.Sequential(*modules)
        return encoder

    def forward(self, x):
        """ Forward pass """
        y = self.encoder(x)
        return y


#