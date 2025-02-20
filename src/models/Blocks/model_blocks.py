"""
Basic building blocks for neural nets
"""

import math
import torch
import torch.nn as nn

from models.Blocks.model_utils import build_grid


__all__ = [
        "ConvBlock",
        "ConvTransposeBlock",
        "SoftPositionEmbed",
        "SlotPositionalEncoding",
        "TemporalPositionalEncoding",
        "LearnedPositionalEncoding1D"
    ]



class Upsample(nn.Module):
    """
    Overriding the upsample class to avoid an error of nn.Upsample with large tensors
    """

    def __init__(self, scale_factor):
        """ Module initializer """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """ Forward pass """
        y = nn.functional.interpolate(
                x.contiguous(),
                scale_factor=self.scale_factor,
                mode='nearest'
            )
        return y

    def __repr__(self):
        """ Nice printing """
        str = f"Upsample(scale_factor={self.scale_factor})"
        return str
    


class ConvBlock(nn.Module):
    """
    Simple convolutional block for conv. encoders

    Args:
    -----
    in_channels: int
        Number of channels in the input feature maps.
    out_channels: int
        Number of convolutional kernels in the conv layer
    kernel_size: int
        Size of the kernel for the conv layer
    stride: int
        Amount of strid applied in the convolution
    padding: int/None
        Whether to pad the input feature maps, and how much padding to use.
    batch_norm: bool
        If True, Batch Norm is applied after the convolutional layer
    max_pool: int/tuple/None
        If not None, output feature maps are downsampled by this amount via max pooling
    activation: bool
        If True, output feature maps are activated via a ReLU nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 batch_norm=False, max_pool=None, activation=True):
        """
        Module initializer
        """
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        # adding conv-(bn)-(pool)-act layer
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if max_pool:
            assert isinstance(max_pool, (int, tuple, list))
            layers.append(nn.MaxPool2d(kernel_size=max_pool, stride=max_pool))
        if activation:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        """
        Forward pass
        """
        y = self.block(x)
        return y



class ConvTransposeBlock(nn.Module):
    """
    Simple transposed-convolutional block for conv. decoders

    Args:
    -----
    in_channels: int
        Number of channels in the input feature maps.
    out_channels: int
        Number of convolutional kernels in the conv layer
    kernel_size: int
        Size of the kernel for the conv layer
    stride: int
        Amount of strid applied in the convolution
    padding: int/None
        Whether to pad the input feature maps, and how much padding to use.
    batch_norm: bool
        If True, Batch Norm is applied after the convolutional layer
    upsample: int/tuple/None
        If not None, output feature maps are upsampled by this amount via (nn.) Upsampling
    activation: bool
        If True, output feature maps are activated via a ReLU nonlinearity.
    conv_transpose_2d: bool
        If True, Transposed convolutional layers are used.
        Otherwise, standard convolutions (combined with Upsampling) are applied.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 batch_norm=False, upsample=None, activation=True, conv_transpose_2d=True):
        """ Module initializer """
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        # adding conv-(bn)-(pool)-act layer
        layers = []
        if conv_transpose_2d:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
            )
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if upsample:
            assert isinstance(upsample, (int, tuple, list))
            layers.append(nn.Upsample(scale_factor=upsample))
        if activation:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        """
        Forward pass
        """
        y = self.block(x)
        return y



class SoftPositionEmbed(nn.Module):
    """
    Soft positional embedding with learnable linear projection.
        1. The positional encoding corresponds to a 4-channel grid
           with coords [-1, ..., 1] and [1, ..., -1] in the vertical
           and horizontal directions
        2. The 4 channels are projected into a hidden_dimension via a
           linear layer (or Conv-1D)


    Args:
    -----
    hidden_size: int
        Number of output channels
    resolution: list/tuple of integers
        Number of elements in the positional embedding. Corresponds to a spatial size
    vmin, vmax: int
        Minimum and maximum values in the grids. By default vmin=-1 and vmax=1
    """

    def __init__(self, hidden_size, resolution, vmin=-1., vmax=1.):
        """
        Soft positional encoding
        """
        super().__init__()
        self.projection = nn.Conv2d(4, hidden_size, kernel_size=1)
        self.grid = build_grid(resolution, vmin=-1., vmax=1.).permute(0, 3, 1, 2)
        return

    def forward(self, inputs, channels_last=True):
        """
        Projecting grid and adding to inputs
        """
        b_size = inputs.shape[0]
        if self.grid.device != inputs.device:
            self.grid = self.grid.to(inputs.device)
        grid = self.grid.repeat(b_size, 1, 1, 1)
        emb_proj = self.projection(grid)
        if channels_last:
            emb_proj = emb_proj.permute(0, 2, 3, 1)
        return inputs + emb_proj



class SlotPositionalEncoding(nn.Module):
    """
    Positional encoding to be added to the input tokens of the transformer predictor.

    Our positional encoding only informs about the time-step, i.e., all slots extracted
    from the same input frame share the same positional embedding.
    This allows our predictor model to maintain the permutation equivariance properties.

    Args:
    -----
    batch_size: int
        Number of elements in the batch.
    num_slots: int
        Number of slots extracted per frame.
        Positional encoding will be repeat for each of these.
    d_model: int
        Dimensionality of the slots/tokens
    dropout: float
        Percentage of dropout to apply after adding the poisitional encoding. Default is 0.1
    max_len: int
        Length of the sequence.
    """

    def __init__(self, d_model, dropout=0.1, max_len=50):
        """
        Initializing the positional encoding
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # initializing sinusoidal positional embedding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, max_len, 1, d_model)
        self.pe = pe
        return

    def forward(self, x, batch_size, num_slots):
        """
        Adding the positional encoding to the input tokens of the transformer

        Args:
        -----
        x: torch Tensor
            Tokens to enhance with positional encoding.
            Shape is (B, Seq_len, Num_Slots, Token_Dim)
        batch_size: int
            Given batch size to repeat the positional encoding for
        num_slots: int
            Number of slots to repear the positional encoder for
        """
        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        cur_seq_len = x.shape[1]
        cur_pe = self.pe.repeat(batch_size, 1, num_slots, 1)[:, :cur_seq_len]
        x = x + cur_pe
        y = self.dropout(x)
        return y



class TemporalPositionalEncoding(nn.Module):
    """
    Positional encoding to be added to the input tokens of the transformer predictor.

    Our positional encoding only informs about the time-step, i.e., all slots extracted
    from the same input frame share the same positional embedding. This allows our predictor
    model to maintain the permutation equivariance properties.

    Args:
    -----
    batch_size: int
        Number of elements in the batch.
    num_slots: int
        Number of slots extracted per frame.
        Positional encoding will be repeat for each of these.
    d_model: int
        Dimensionality of the slots/tokens
    dropout: float
        Percentage of dropout to apply after adding the poisitional encoding. Default is 0.1
    max_len: int
        Length of the sequence.
    mode: string
        Type of positional encoding to use. It can be one of ['sinusoid', 'learned']
    """

    MODES = ["sinusoid", "learned"]

    def __init__(self, d_model, dropout=0.0, max_len=50, mode="sinusoid"):
        """
        Initializing the positional encoding
        """
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown {mode = }. Use one of {self.MODES}...")
        self.mode = mode
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=dropout)

        # initializing embedding
        self.pe = self._get_pe()
        return

    def _get_pe(self):
        """
        Initializing the temporal positional encoding given the encoding mode
        """
        max_len = self.max_len
        d_model = self.d_model
        if self.mode == "sinusoid":
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            pe = pe.view(1, max_len, 1, d_model)
        elif self.mode == "learned":
            scale = d_model ** -0.5
            pe = nn.Parameter(scale * torch.randn(1, max_len, 1, d_model))
        else:
            raise ValueError(f"UPSI! Unknown {self.mode = }. Use one of {self.MODES}...")
        return pe

    def forward(self, x, batch_size, num_slots):
        """
        Adding the positional encoding to the input tokens of the transformer

        Args:
        -----
        x: torch Tensor
            Tokens to enhance with positional encoding.
            Shape is (B, Seq_len, Num_Slots, Token_Dim)
        batch_size: int
            Given batch size to repeat the positional encoding for
        num_slots: int
            Number of slots to repear the positional encoder for
        """
        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        cur_seq_len = x.shape[1]
        cur_pe = self.pe.repeat(batch_size, 1, num_slots, 1)[:, :cur_seq_len]
        cur_pe = torch.flip(cur_pe, dims=(1,))
        x = x + cur_pe
        y = self.dropout(x)
        return y

    def __repr__(self):
        """ String represetnation """
        str = ""
        str += "TemporalPositionalEncoding(\n"
        str += f"  (mode) {self.mode})\n"
        str += f"  (max_len) {self.max_len})\n"
        str += f"  (d_model) {self.d_model})\n"
        str += f"  (dropout): Dropout(p={self.dropout_p} inplace=False)\n"
        str += ")"
        return str



class LearnedPositionalEncoding1D(nn.Module):
    """
    1-Dimensional learned positional encoding
    """

    def __init__(self, max_len, token_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, token_dim), requires_grad=True)
        nn.init.trunc_normal_(self.pe)
        return

    def forward(self, x, offset=0):
        """ Adding positional encoding """
        T = x.shape[1] # seq_len
        y = self.dropout(x + self.pe[:, offset:offset + T])
        return y

