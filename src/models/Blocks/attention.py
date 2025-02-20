"""
Attention modules:
"""

import torch
import torch.nn as nn

from models.Blocks.model_utils import init_xavier_



class SlotAttention(nn.Module):
    """
    Implementation of the SlotAttention module from:
      --> Locatello, et al. "Object-centric learning with slot attention." NeurIPS 2020

    Args:
    -----
    dim_feats: integer
        Dimensionality of the input embeddings
    dim_slots: integer
        Dimensionality of the object slots
    Num_slots: integer
        Number of slots competing for representing the image
    num_iters_first: integer
        Number of recurrent iterations to refine the slots for the first video frame.
    num_iters: integer
        Number of recurrent iterations to refine the slots from the second frame onwards.
    mlp_hidden_size: integer
        Hidden dimensionality of the mlp,
    epsilon: float
        Small value used to stabilize divisiona and softmax
    """

    def __init__(self, dim_feats, dim_slots, num_slots, num_iters_first=2, num_iters=2,
                 mlp_hidden=128, epsilon=1e-8):
        """
        Module Initializer
        """
        super().__init__()
        self.dim_slots = dim_slots
        self.num_iters_first = num_iters_first
        self.num_iters = num_iters
        self.num_slots = num_slots
        self.epsilon = epsilon
        self.scale = dim_feats ** -0.5

        # normalization layers
        self.norm_input = nn.LayerNorm(dim_feats, eps=0.001)
        self.norm_slot = nn.LayerNorm(dim_slots, eps=0.001)
        self.norm_mlp = nn.LayerNorm(dim_slots, eps=0.001)

        # attention embedders
        self.to_q = nn.Linear(dim_slots, dim_slots)
        self.to_k = nn.Linear(dim_feats, dim_slots)
        self.to_v = nn.Linear(dim_feats, dim_slots)

        # Slot update functions.
        self.gru = nn.GRUCell(dim_slots, dim_slots)
        self.mlp = nn.Sequential(
                nn.Linear(dim_slots, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, dim_slots),
            )
        return

    def forward(self, inputs, slots, step=0, **kwargs):
        """
        Forward pass as depicted in Algorithm 1 from paper

        Args:
        -----
        inputs: torch Tensor
            input feature vectors extracted by the encoder.
            Shape is (Batch, Num locations, Dimensionality)

        Returns:
        --------
        slots: torch Tensor
            Slot assignment for each of the input vectors
            Shape is (Batch, Num Slots, Slot Dimensionality)
        """
        B = inputs.shape[0]
        self.attention_masks = None

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        # iterative refinement of the slot representation
        num_iters = self.num_iters_first if step == 0 else self.num_iters
        for _ in range(num_iters):
            slots_prev = slots
            slots = self.norm_slot(slots)
            q = self.to_q(slots)

            # q ~ (B, N_Slots, Slot_dim)
            # k, v ~ (B, N_locs, Slot_dim)
            # attention equation [softmax(Q K^T) V]
            dots = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale 
            attn = dots.softmax(dim=1) + self.epsilon 
            self.attention_masks = attn
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('b i d , b d j -> b i j', attn, v)
            # further refinement
            slots = self.gru(
                    updates.reshape(-1, self.dim_slots),
                    slots_prev.reshape(-1, self.dim_slots)
                )
            slots = slots.reshape(B, -1, self.dim_slots)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

    def get_attention_masks(self, shape=None):
        """
        Fetching last computer attention masks

        Returns:
        --------
        attention_masks: torch Tensor
            attention masks highligtinh the importance of each location to each slot
            Shape is (B, N_slots, N_locs)
        """
        B, N_slots, _ = self.attention_masks.shape
        masks = self.attention_masks
        if shape is not None:
            masks = masks.reshape(B, N_slots, *shape)
        return masks


########################
# ATTENTION MECHANISMS #
########################


class MetaAttention(nn.Module):
    """
    MetaClass for (Multi-Head) Key-Value Attention Mechanisms

    Args:
    -----
    emb_dim: integer
        Dimensionality of the token embeddings.
    num_heads: integer
        Number of heads accross which we compute attention.
        Head_dim = Emb_dim / Num_Heads. Division must be exact!
    dropout: float [0,1]
        Percentage of connections dropped during the attention
    out_dim: int/None
        Dimensionality of the output embeddings. If not given, it is set to 'emb_dim'
    """

    def __init__(self, emb_dim, num_heads=1, dropout=0., out_dim=None, **kwargs):
        """
        Initializer of the attention block
        """
        assert num_heads >= 1
        if emb_dim % num_heads != 0:
            raise ValueError(f"{emb_dim = } must be divisible by {num_heads}...")
        super().__init__()

        out_dim = out_dim if out_dim is not None else emb_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        # computing query, key, value for all embedding heads
        self.q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

        # output projection
        self.out_projection = nn.Sequential(
                nn.Linear(emb_dim, out_dim, bias=False)
            )
        self.attention_masks = None
        return

    def forward(self, x):
        """ """
        raise NotImplementedError("Base-Class does not implement a 'forward' method...")

    def attention(self, query, key, value, dim_head, mask=None, **kwargs):
        """
        Implementation of the standard normalized key-value attention equation
        """
        scale = dim_head ** -0.5
        dots = torch.einsum('b i d , b j d -> b i j', query, key) * scale
        if mask is not None:
            dots = dots.masked_fill(mask, float('-inf'))
        attention = dots.softmax(dim=-1)
        attention = self.drop(attention)
        vect = torch.einsum('b i d , b d j -> b i j', attention, value)        
        return vect

    def split_into_heads(self, x):
        """
        Splitting a vector into multiple heads
        """
        batch_size, num_tokens, token_dim = x.shape

        dim_head = token_dim // self.num_heads

        x = x.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        y = x.reshape(batch_size * self.num_heads, num_tokens, dim_head)
        return y

    def merge_heads(self, x):
        """
        Rearranging heads and recovering original shape
        """
        _, num_tokens, dim_head = x.shape
        x = x.reshape(-1, self.num_heads, num_tokens, dim_head).transpose(1, 2)
        y = x.reshape(-1, num_tokens, self.num_heads * dim_head)
        return y



class MultiHeadSelfAttention(MetaAttention):
    """
    Vanilla Multi-Head dot-product attention mechanism.

    Args:
    -----
    emb_dim: integer
        Dimensionality of the token embeddings.
    num_heads: integer
        Number of heads accross which we compute attention.
        Head_dim = Emb_dim / Num_Heads. Division must be exact!
    dropout: float [0,1]
        Percentage of connections dropped during the attention
    """

    def __init__(self, emb_dim, num_heads=8, dropout=0.):
        """
        Initializer of the attention block
        """
        super().__init__(
                emb_dim=emb_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        return

    def forward(self, x, **kwargs):
        """
        Forward pass through multi-head attention
        """
        # linear projections
        # linear projections and splitting into heads:
        # (B, N, D) --> (B, N, Nh, Dh) --> (B * Nh, N, Dh)
        dim_head = x.shape[-1] // self.num_heads
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # applying attention equation
        mask = kwargs.get("mask", None)
        vect = self.attention(query=q, key=k, value=v, dim_head=dim_head, mask=mask)

        # rearranging heads and recovering shape:
        y = self.merge_heads(vect)
        y = self.out_projection(y)
        return y



class MultiHeadCrossAttention(MetaAttention):
    """
    Multi-Head cross-product attention mechanism, as uses in a Transformer decoder.

    Args:
    -----
    emb_dim: integer
        Dimensionality of the token embeddings.
    num_heads: integer
        Number of heads accross which we compute attention.
        Head_dim = Emb_dim / Num_Heads. Division must be exact!
    dropout: float [0,1]
        Percentage of connections dropped during the attention
    """

    def __init__(self, emb_dim, dim_head, kv_dim, num_heads=8, dropout=0.):
        """
        Initializer of the attention block
        """
        super().__init__(
                emb_dim=emb_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        self.dim_head = dim_head
        
        inner_dim = dim_head * num_heads
        self.q = nn.Linear(emb_dim, inner_dim, bias=False)
        self.k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.v = nn.Linear(kv_dim, inner_dim, bias=False)

        self.out_projection = nn.Linear(inner_dim, emb_dim)
        return

    def forward(self, enc_embs, query_embs, **kwargs):
        """
        Forward pass through multi-head self-attention
        """
        # linear projections and splitting into heads
        q, k, v = self.q(query_embs), self.k(enc_embs), self.v(enc_embs)
        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # applying attention equation
        vect = self.attention(query=q, key=k, value=v, dim_head=self.dim_head)

        # rearranging heads and recovering original shape
        y = self.merge_heads(vect)
        y = self.out_projection(y)
        return y



class TransformerBlock(nn.Module):
    """
    Tranformer encoder block.
    This is used as predictor module in SAVi.

    Args:
    -----
    embed_dim: int
        Dimensionality of the input embeddings
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    """

    def __init__(self, embed_dim, num_heads, mlp_size, pre_norm=True):
        """
        Module initializer
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        assert num_heads >= 1

        # MHA
        self.attn = MultiHeadSelfAttention(
            emb_dim=embed_dim,
            num_heads=num_heads,
        )
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_dim),
        )
        # LayerNorms
        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_model()
        return

    @torch.no_grad()
    def _init_model(self):
        """ Parameter initialization """
        init_xavier_(self)

    def forward(self, inputs):
        """
        Forward pass through transformer encoder block
        """
        assert inputs.ndim == 3
        # pre-norm
        if self.pre_norm:
            # Self-attention.
            x = self.layernorm_query(inputs)
            x = self.attn(x)
            y = x + inputs
            # MLP
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            z = z + y
        # post-norm
        else:
            # Self-attention.
            x = self.attn(inputs)
            y = x + inputs
            y = self.layernorm_query(y)
            # MLP
            z = self.mlp(y)
            z = z + y
            z = self.layernorm_mlp(z)
        return z



class TransformerDecoderBlock(nn.Module):
    """
    Tranformer decoder block with cross-attention only

    Args:
    -----
    embed_dim: int
        Dimensionality of the input embeddings
    head_dim: int
        Dimensionality of each of the attention heads
    kv_dim: int
        Dimensionality of the input features used as keys and values
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    pre_norm: bool
        If True, transformer computes the LayerNorm before attention and MLP.
        Otherwise, LayerNorm is used after the aforementaitoned layers
    """

    def __init__(self, embed_dim, head_dim, kv_dim, num_heads, mlp_size):
        """
        Module initializer
        """
        super().__init__()

        self.ln_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_dim),
        )

        # Cross Attention
        self.ln_cross_att_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_cross_att_kv = nn.LayerNorm(kv_dim, eps=1e-6)
        self.cross_attn = MultiHeadCrossAttention(
                emb_dim=embed_dim,
                dim_head=head_dim,
                num_heads=num_heads,
                kv_dim=kv_dim
            )
        return

    def forward(self, queries, feats):
        """
        Forward pass through transformer encoder block
        """
        assert queries.ndim == 3
        B, L, _ = queries.shape

        # Cross-attention
        query_embs = self.ln_cross_att_q(queries)
        feats = self.ln_cross_att_kv(feats)
        z = self.cross_attn(feats, query_embs=query_embs)
        z = z + queries

        # MLP
        out = self.ln_mlp(z)
        out = self.mlp(out)
        out = out + z

        return out

    def get_attention_masks(self, reshape=None):
        """ Fetching last computed attention masks """
        return self.cross_attn.get_attention_masks(reshape=reshape)



class AdaptedEncoderBlock(TransformerBlock):
    """
    Tranformer block with Cross-Attention for Text-to-Slot Attention

    Args:
    -----
    embed_dim: int
        Dimensionality of the input embeddings
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    """

    def __init__(self, embed_dim, num_heads, mlp_size, fusion_params):
        """
        Module initializer
        """
        super().__init__(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
            )

        self.cross_attention = TransformerDecoderBlock(
                embed_dim=embed_dim,
                kv_dim=embed_dim,
                head_dim=fusion_params.get("head_dim"),
                num_heads=fusion_params.get("num_heads"),
                mlp_size=fusion_params.get("mlp_size")
            )
        return

    def forward(self, x, text_embeddings):
        """
        Override the forward pass through transformer block to process the object slots
        and perform text-to-slot cross-attention between slots and caption text
        """
        assert x.ndim == 3, f"Input 'x' must have 3 dims, but got {x.shape = }..."

        # Self-attention.
        y = self.layernorm_query(x)
        y = self.attn(y)
        y = y + x
        # Cross-attention.
        z = self.condition_slots_given_caption(
                slots_to_condition=y,
                text_embeddings=text_embeddings
            )
        # MLP
        z = self.layernorm_mlp(z)
        z = self.mlp(z)
        out = z + y
        return out

    def condition_slots_given_caption(self, slots_to_condition, text_embeddings):
        """
        Conditioning a set of object-slots given a textual caption
        """
        conditioned_slots = self.cross_attention(
                queries=slots_to_condition,
                feats=text_embeddings
            )
        return conditioned_slots


