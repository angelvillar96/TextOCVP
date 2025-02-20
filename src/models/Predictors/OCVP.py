"""
Implementation of uncoditioned predictor modules.
Code adapted from:
  - https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction/tree/master 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.logger import print_

from models.Blocks.model_blocks import SlotPositionalEncoding
from models.Blocks.model_utils import init_kaiming_


__all__ = [
        "VanillaTransformerPredictor",
        "OCVPSeq",
        "OCVPPar",
    ]



class VanillaTransformerPredictor(nn.Module):
    """
    Vanilla Transformer Predictor module.
    It performs self-attention over all slots in the input buffer,
    jointly modelling the relational and temporal dimensions.

    Args:
    -----
    num_slots: int
        Number of slots per image.
        Num. inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    token_dim: int
        Input slots are mapped to this dimensionality via a linear layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, token_dim=128, hidden_dim=256,
                 num_layers=2, n_heads=4, residual=False, input_buffer_size=5):
        """
        Module initializer
        """
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        self.input_buffer_size = input_buffer_size
        print_("Instanciating Vanilla Transformer Predictor:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.nhead}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")
        print_(f"  --> input_buffer_size: {self.input_buffer_size}")

        # MLPs to map slot-dim into token-dim and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # Transformer Encoder
        self.transformer_encoders = nn.Sequential(
            *[torch.nn.TransformerEncoderLayer(
                    d_model=token_dim,
                    nhead=self.nhead,
                    batch_first=True,
                    norm_first=True,
                    dim_feedforward=hidden_dim
                ) for _ in range(num_layers)]
            )

        # Custom positional encoding. All slots from the same time step share the token
        self.pe = SlotPositionalEncoding(
                d_model=self.token_dim,
                max_len=input_buffer_size
            )
        self._init_model()
        return

    def forward(self, slots, **kwargs):
        """
        Foward pass through the transformer predictor module to predict
        the subsequent object slots
        Args:
        -----
        slots: torch Tensor
            Input object slots from the previous time steps.
            Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots.
            Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_imgs, num_slots, _ = slots.shape

        # mapping slots to tokens, and applying temporal positional encoding
        token_input = self.mlp_in(slots)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding through transformer encoder blocks
        token_output = time_encoded_input.reshape(B, num_imgs * num_slots, self.token_dim)
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)
        token_output = token_output.reshape(B, num_imgs, num_slots, self.token_dim)

        # mapping back to slot dimensionality
        output = self.mlp_out(token_output[:, -1])
        output = output + slots[:, -1] if self.residual else output
        return output
    
    @torch.no_grad()
    def _init_model(self):
        """ Parameter initialization """
        init_kaiming_(self)
        return



class OCVPSeq(nn.Module):
    """
    Sequential Object-Centric Video Prediction Transformer Module (OCVP-Seq).
    This module models the temporal dynamics and object interactions in a
    decoupled manner by sequentially applying object- and time-attention
    i.e. [time, obj, time, ...]

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is n_slots * n_imgs
    slot_dim: int
        Dimensionality of the input slots
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, token_dim=128, hidden_dim=256, num_layers=2,
                 n_heads=4, residual=False, input_buffer_size=5):
        """
        Module Initialzer
        """
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        self.input_buffer_size = input_buffer_size

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        print_("Instanciating OCVP-Seq Predictor Module:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.nhead}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")
        print_(f"  --> input_buffer_size: {self.input_buffer_size}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        self.transformer_encoders = nn.Sequential(
            *[OCVPSeqLayer(
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads
                ) for _ in range(num_layers)]
            )

        # custom temporal encoding.
        # All slots from the same time step share the same encoding
        self.pe = SlotPositionalEncoding(
                d_model=self.token_dim,
                max_len=input_buffer_size
            )
        return

    def forward(self, slots, **kwargs):
        """
        Forward pass through OCVP-Seq

        Args:
        -----
        slots: torch Tensor
            Input object slots from the previous time steps.
            Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim),
            but we only care about the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, _, num_slots, _ = slots.shape

        # projecting slots into tokens, and applying positional encoding
        token_input = self.mlp_in(slots)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding through OCVP-Seq transformer blocks
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        # mapping back to the slot dimension
        output = self.mlp_out(token_output[:, -1])
        output = output + slots[:, -1] if self.residual else output
        return output



class OCVPSeqLayer(nn.Module):
    """
    Sequential Object-Centric Video Prediction (OCVP-Seq) Transformer Layer.
    Sequentially applies object- and time-attention.

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    hidden_dim: int
        Hidden dimensionality of the MLPs in the transformer modules
    n_heads: int
        Number of heads for multi-head self-attention.
    """

    def __init__(self, token_dim=128, hidden_dim=256, n_heads=4):
        """
        Module initializer
        """
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.nhead = n_heads

        self.object_encoder_block = torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            )
        self.time_encoder_block = torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            )
        return

    def forward(self, inputs, time_mask=None):
        """
        Forward pass through the Object-Centric Transformer-V1 Layer

        Args:
        -----
        inputs: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        """
        B, num_imgs, num_slots, dim = inputs.shape

        # object-attention block. Operates on (B * N_imgs, N_slots, Dim)
        inputs = inputs.reshape(B * num_imgs, num_slots, dim)
        object_encoded_out = self.object_encoder_block(inputs)
        object_encoded_out = object_encoded_out.reshape(B, num_imgs, num_slots, dim)

        # time-attention block. Operates on (B * N_slots, N_imgs, Dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        object_encoded_out = object_encoded_out.reshape(B * num_slots, num_imgs, dim)
        object_encoded_out = self.time_encoder_block(object_encoded_out)
        object_encoded_out = object_encoded_out.reshape(B, num_slots, num_imgs, dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        return object_encoded_out



class OCVPPar(nn.Module):
    """
    Parallel Object-Centric Video Prediction (OCVP-Par) Transformer Predictor Module.
    This module models the temporal dynamics and object interactions in a
    dissentangled manner by applying relational- and temporal-attention in parallel.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is n_slots * n_imgs
    slot_dim: int
        Dimensionality of the input slots
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, token_dim=128, hidden_dim=256, num_layers=2,
                 n_heads=4, residual=False, input_buffer_size=5):
        """
        Module initializer
        """
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        self.input_buffer_size = input_buffer_size

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        print_("Instanciating OCVP-Par Predictor Module:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.nhead}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")
        print_(f"  --> input_buffer_size: {self.input_buffer_size}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        self.transformer_encoders = nn.Sequential(
            *[OCVPParLayer(
                    d_model=token_dim,
                    nhead=self.nhead,
                    batch_first=True,
                    norm_first=True,
                    dim_feedforward=hidden_dim
                ) for _ in range(num_layers)]
            )

        self.pe = SlotPositionalEncoding(
                d_model=self.token_dim,
                max_len=input_buffer_size
            )
        return

    def forward(self, slots, **kwargs):
        """
        Forward pass through Object-Centric Transformer v1

        Args:
        -----
        slots: torch Tensor
            Input object slots from the previous time steps.
            Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots.
            Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, _, num_slots, _ = slots.shape

        # projecting slots and applying positional encodings
        token_input = self.mlp_in(slots)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding tokens through transformer la<ers
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        # projecting back to slot-dimensionality
        output = self.mlp_out(token_output[:, -1])
        output = output + slots[:, -1] if self.residual else output
        return output



class OCVPParLayer(nn.TransformerEncoderLayer):
    """
    Parallel Object-Centric Video Prediction (OCVP-Par) Transformer Module.
    This module models the temporal dynamics and object interactions in a
    dissentangled manner by applying object- and time-attention in parallel.

    Args:
    -----
    d_model: int
        Dimensionality of the input tokens
    nhead: int
        Number of heads in multi-head attention
    dim_feedforward: int
        Hidden dimension in the MLP
    dropout: float
        Amount of dropout to apply. Default is 0.1
    activation: int
        Nonlinear activation in the MLP. Default is ReLU
    layer_norm_eps: int
        Epsilon value in the layer normalization components
    batch_first: int
        If True, shape is (B, num_tokens, token_dim);
        otherwise, it is (num_tokens, B, token_dim)
    norm_first: int
        If True, transformer is in mode pre-norm: otherwise, it is post-norm
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=True,
                 norm_first=True, device=None, dtype=None):
        """
        Module initializer
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                device=device,
                dtype=dtype
            )

        self.self_attn_obj = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=batch_first,
                **factory_kwargs
            )
        self.self_attn_time = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=batch_first,
                **factory_kwargs
            )
        return

    def forward(self, src, time_mask=None):
        """
        Forward pass through the Object-Centric Transformer-v2.
        Overloads PyTorch's transformer forward pass.

        Args:
        -----
        src: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), time_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, time_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, time_mask):
        """
        Forward pass through the parallel attention branches
        """
        B, num_imgs, num_slots, dim = x.shape

        # object-attention
        x_aux = x.clone().view(B * num_imgs, num_slots, dim)
        x_obj = self.self_attn_obj(
                query=x_aux,
                key=x_aux,
                value=x_aux,
                need_weights=False
            )[0]
        x_obj = x_obj.view(B, num_imgs, num_slots, dim)

        # time-attention
        x = x.transpose(1, 2).reshape(B * num_slots, num_imgs, dim)
        x_time = self.self_attn_time(
                query=x,
                key=x,
                value=x,
                attn_mask=time_mask,
                need_weights=False
            )[0]
        x_time = x_time.view(B, num_slots, num_imgs, dim).transpose(1, 2)

        y = self.dropout1(x_obj + x_time)
        return y


