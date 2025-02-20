"""
Modules for text encoding and tokenization
"""

import numpy as np
import nltk
import torch
import torch.nn as nn

nltk.download('punkt')



class TransformerTextEncoder(nn.Module):
    """
    Transformer-based text-encoder.
    Maps the input text sequence into text embeddings. Each word is mapped into a single embedding.

    Args:
    -----
    input_dim: int
        Dimensionality of the input word embeddings
    num_layers: int
        Number of transformer encoder layers to have in the encoder
    num_heads: int
        Number of self-attention heads in each encoder layer
    output_dim: int
        Dimensionality of the output embeddings
    vocab_size: int
        Number of words in the lookup table mapping words to embeddings.
    context_length: int
        Length of the positional embeddings added to the text embeddings.
    """

    def __init__(self, input_dim, num_layers, num_heads, output_dim, vocab_size,
                 context_length=50, dropout=0.1):
        """
        Module initializer
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = 0

        # building transformer encoder module
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # other require layers and modules
        self.token_embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=input_dim
            )
        self.position_embedding = nn.Embedding(
                num_embeddings=context_length,
                embedding_dim=input_dim
            )
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-8, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)
        self.text_out_projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim)
            )

        self.apply(self._init_weights)
        return

    @staticmethod
    def _init_weights(module):
        """
        Weight initialization in this module
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        return

    def forward(self, text, text_length):
        """
        Forward pass through the transformer text encoder

        Args:
        -----
        text: torch Tensor
            Input text tokens
        text_length: torch Tensor
            Original number of words in the caption.
            It must be given in order to stack them in a batch,
            since the captions might have been padded 
        """
        # computing text tokens and adding positional encodings
        position_indices = self._create_position_indices(text)
        text_tokens = self.token_embedding(text)
        position_embeddings = self.position_embedding(position_indices)
        text_tokens = self.layer_norm(text_tokens + position_embeddings)
        text_tokens = self.dropout(text_tokens)

        # computing masks for padding tokens.
        token_mask = (text != self.padding_idx).unsqueeze(-1)
        text_tokens = text_tokens * token_mask.type(text_tokens.dtype)
        ones = torch.ones_like(text)
        caption_mask = text_length.unsqueeze(1) < ones.cumsum(dim=1)

        # encoding text tokens
        text_tokens = text_tokens.permute(1, 0, 2)  # NLD -> LND
        text_embeddings = self.transformer(
                text_tokens,
                mask=None,
                src_key_padding_mask=caption_mask,
            )
        text_embeddings = text_embeddings.permute(1, 0, 2)  # LND -> NLD

        text_embeddings = self.text_out_projection(text_embeddings)
        return text_embeddings

    def _create_position_indices(self, tokens):
        """
        Creating position indices of the same size as the text tokens.
        """
        batch_size, max_caption_length = tokens.size()
        positions = torch.arange(
                max_caption_length,
                dtype=tokens.dtype,
                device=tokens.device
            )
        positions = positions.view(1, max_caption_length).repeat(batch_size, 1)
        return positions



class CustomTokenizer:
    """
    Custom tokenizer
    """
    
    def __init__(self, vocabulary):
        """ Tokenizer initialization. """
        assert '[PAD]' in vocabulary, "Vocabulary must contain '[PAD]' token..."
        self.padding_idx = vocabulary['[PAD]']
        self.vocabulary = vocabulary
        self.vocabulary_reverse = {v: k for k, v in self.vocabulary.items()}
        return
    
    def tokenize(self, caption):
        """ Tokenizing a single caption """
        caption_tokens = self.text2tokens(caption)
        caption_tokens = np.insert(caption_tokens, 0, self.vocabulary['[CLS]'])
        caption_tokens = np.append(caption_tokens, self.vocabulary['[SEP]'])
        caption_tokens = torch.tensor(caption_tokens, dtype=torch.long)
        caption_length = torch.tensor(len(caption_tokens), dtype=torch.long)
        return caption_tokens, caption_length 

    def tokenize_batch(self, caption):
        """ Tokenizing a batch of captions """        
        caption_tokens, caption_lengths = [], []
        for capt in caption:
            cur_tokens, cur_length = self.tokenize(capt)
            caption_tokens.append(cur_tokens)
            caption_lengths.append(cur_length)
        caption_tokens = torch.nn.utils.rnn.pad_sequence(
                caption_tokens,
                batch_first=True,
                padding_value=self.padding_idx,
            )
        caption_lengths = torch.stack(caption_lengths)
        return caption_tokens, caption_lengths


    def text2tokens(self, x):
        """ Mapping words to tokens """
        words = nltk.word_tokenize(x)
        m = np.int32(np.zeros((1, len(words))))

        for i in range(len(words)):
            m[0, i] = self.vocabulary[words[i]]
        return m

    def tokens2text(self, tokens):
        """ Mapping from tokens to words """
        text = ""
        for i in range(tokens.shape[0]):
            text = text + " " + self.vocabulary_reverse[tokens[i].item()]
        return text

