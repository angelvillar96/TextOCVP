"""
Implementation of encoder modules and factory method for CNN and
transformer-based encoders.
"""

from functools import partial
import torch
import torch.nn as nn

import timm
from timm.models import layers, resnet
from timm.models.vision_transformer import _create_vision_transformer, VisionTransformer

from models.Blocks.model_utils import freeze_params



class ViTEncoder(nn.Module):
    """
    Wrapper for ViT-based encoder modules, which are loaded from 'timm'.
    It removes the classification head, as well as the 'class' token from the output.

    Args:
    -----
    vit_backbone: VisionTransformer
        Pretrained ViT backbone instanciated from 'timm'
    """

    def __init__(self, vit_backbone, num_blocks=None):
        """
        ViT wrapper initializer
        """
        if not isinstance(vit_backbone, VisionTransformer):
            raise TypeError(f"ViT must be a 'timm' VisionTransfromer")
        if not hasattr(vit_backbone, "blocks"):
            raise ValueError(f"ViT backbone must have attribute 'blocks'")
        if num_blocks is not None:
            if len(vit_backbone.blocks) < num_blocks or num_blocks < 0:
                raise ValueError(
                        f"{num_blocks =} must be in [0, {len(vit_backbone.blocks)}]"
                    )
        super().__init__()
        self.vit_backbone = vit_backbone
        self.num_blocks = num_blocks
        
        # getting only required transformer blocks
        if num_blocks is not None:
            self.vit_backbone.blocks = self.vit_backbone.blocks[:num_blocks]            
        freeze_params(self)
        
        self.mean = torch.tensor(
                self.vit_backbone.default_cfg["mean"]).view(1, 1, 3, 1, 1
            )
        self.std = torch.tensor(
                self.vit_backbone.default_cfg["mean"]
            ).view(1, 1, 3, 1, 1)
        return
        
    def forward(self, x):
        """
        Forward pass of a batch of images
        """
        x = self.normalize_images(x)
        x = self.vit_backbone.patch_embed(x)
        x = self.vit_backbone._pos_embed(x)
        x = self.vit_backbone.patch_drop(x)
        x = self.vit_backbone.norm_pre(x)
        x = self.vit_backbone.blocks(x)
        x = x[:, 1:]  # removing class patch
        return x

    @torch.no_grad()
    def _get_num_patches(self):
        """
        Computing the number of patches used by the backbone by forwarding a dummy tensor
        """
        dummy = torch.randn(1, 3, 224, 224)
        out = self.forward(dummy)
        num_patches = out.shape[1]
        return num_patches
    
    def normalize_images(self, img):
        """ 
        Normalizing the input images given the ImageNet mean and standard deviation
        """
        if self.mean.device != img.device:
            self.mean = self.mean.to(img.device)
            self.std = self.std.to(img.device)
        if len(img.shape) == 4:
            mean, std = self.mean[0], self.std[0]
        elif len(img.shape) == 5:
            mean, std = self.mean, self.std
        else:
            raise ValueError(f"Weird {img.shape = }. It should be either 4- or 5-dim")
        norm_img = (img - mean) / std
        return norm_img



def resnet34_savi(pretrained=True, **kwargs):
    """
    ResNet34 as used in SAVi and SAVi++.
    """
    if pretrained:
        raise ValueError("No pretrained weights available for `savi_resnet34`.")
    model_kwargs = dict(
            block=resnet.BasicBlock,
            layers=[3, 4, 6, 3],
            norm_layer=layers.GroupNorm
            **kwargs
        )
    model = resnet._create_resnet(
            "resnet34",
            pretrained=pretrained,
            **model_kwargs
        )
    model.conv1.stride = (1, 1)
    model.maxpool.stride = (1, 1)
    return model



def vit_small_patch16_224_dino(pretrained=True, **kwargs):
    """
    ViT-small encoder with patch-size 16.
    """
    model_kwargs = dict(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            **kwargs,
        )
    model = _create_vision_transformer(
            "vit_small_patch16_224.dino",
            pretrained=pretrained,
            **model_kwargs
        )
    return model



def vit_small_patch8_224_dino(pretrained=True, **kwargs):
    """
    ViT-small encoder with patch-size 8.
    """
    model_kwargs = dict(
            patch_size=8,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            **kwargs,
        )
    model = _create_vision_transformer(
            "vit_small_patch8_224.dino",
            pretrained=pretrained,
            **model_kwargs
        )
    return model



def vit_base_patch16_224_dino(pretrained=True, **kwargs):
    """
    ViT-base encoder with patch-size 16.
    """
    model_kwargs = dict(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            **kwargs,
        )
    model = _create_vision_transformer(
            "vit_base_patch16_224.dino",
            pretrained=pretrained,
            **model_kwargs
        )
    return model



def vit_base_patch8_224_dino(pretrained=True, **kwargs):
    """
    ViT-base encoder with patch-size 8.
    """
    model_kwargs = dict(
            patch_size=8,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            **kwargs,
        )
    model = _create_vision_transformer(
            "vit_base_patch8_224.dino",
            pretrained=pretrained,
            **model_kwargs
        )
    return model



def vit_small_patch14_dinov2(pretrained=True, **kwargs):
    """
    ViT-small encoder with patch-size 14, pretrained with self-supervised DINOv2 method.

    Image size: 518 x 518
    """
    model_kwargs = dict(
            patch_size=14,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            **kwargs,
        )
    model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            **model_kwargs
        )
    return model



def vit_base_patch14_dinov2(pretrained=True, **kwargs):
    """
    ViT-base encoder with patch-size 14, pretrained with self-supervised DINOv2 method.

    Image size: 518 x 518
    """
    model_kwargs = dict(
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            **kwargs,
        )
    model = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            **model_kwargs
        )
    return model


