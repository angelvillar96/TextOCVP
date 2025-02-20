""" 
Implemenation of Text-Conditioned Object-Centric Video Prediction modules.
"""

import torch.nn as nn

from lib.logger import print_
from models.Blocks.model_utils import freeze_params
from models.Blocks.model_blocks import TemporalPositionalEncoding
from models.EncodersDecoders.text_encoders import TransformerTextEncoder
from models.Blocks.attention import AdaptedEncoderBlock
from transformers import T5EncoderModel


__all__ = [
        "TextOCVP_CustomTF", 
        "TextOCVP_T5", 
]   


class BaseTextOCVP(nn.Module):
    """ 
    Base module for TextOCVP modules.
    These are transformer module that predicts future object slots conditioned
    on past object slots and a text caption that defines the motion and goals.
    """

    def __init__(self, slot_dim, predictor_params, fusion_params, text_encoder_params):
        """
        Module initalizer
        """
        super().__init__()
        self.predictor_params = predictor_params
        self.fusion_params = fusion_params
        self.text_encoder_params = text_encoder_params

        # main predictor parameters
        self.slot_dim = slot_dim
        self.token_dim = predictor_params.get("token_dim") 
        self.num_heads = predictor_params.get("n_heads")
        self.hidden_dim = predictor_params.get("hidden_dim")
        self.num_layers = predictor_params.get("num_layers")
        self.residual = predictor_params.get("residual")
        self.input_buffer_size = predictor_params.get("input_buffer_size")

        # text-conditioned slot predictor module
        self.mlp_in = nn.Linear(self.slot_dim, self.token_dim)
        self.mlp_out = nn.Linear(self.token_dim, self.slot_dim)
        self.predictor = nn.ModuleList(
            [AdaptedEncoderBlock(
                    embed_dim=self.token_dim,
                    num_heads=self.num_heads,
                    mlp_size=self.hidden_dim,
                    fusion_params=self.fusion_params
                ) for _ in range(self.num_layers)]
            )
        self.device = self.mlp_in.bias.device
        
        # Text encoder module
        self._instantiate_text_encoder()

        # Custom temporal encoding.
        self.pe = TemporalPositionalEncoding(
                d_model=self.token_dim,
                max_len=self.input_buffer_size + 1,
                mode="learned"
            )

        self.conditioned_slots = None
        self._log_model()
        return
    
    def _instantiate_text_encoder(self):
        """ Instanciating text encoder """
        raise NotImplementedError(
                f"'BaseTextOCVP' does not implement '_instantiate_text_encoder'..."
            )
        
    def forward(self, slots, text_embeddings, **kwargs):
        """
        Forward pass through text-conditional predictor model
        """
        B, num_imgs, num_slots, _ = slots.shape

        # mapping slots to tokens, and applying temporal positional encoding
        tokens = self.mlp_in(slots)
        token_input = self.pe(
                x=tokens,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding through transformer predictor blocks
        tokens = token_input.reshape(B, num_imgs * num_slots, self.token_dim)
        for pred_block in self.predictor:
            tokens = pred_block(
                    x=tokens,
                    text_embeddings=text_embeddings
                )
        tokens = tokens.reshape(B, num_imgs, num_slots, self.token_dim)

        # mapping back to slot dimensionality
        output = self.mlp_out(tokens[:, -1])
        output = output + slots[:, -1] if self.residual else output
        return output

    def _log_model(self):
        """ Printing parameters of the model """
        print_("Instanciating Text-Conditional Transformer Predictor")
        print_(f"Predictor: {self.__class__.__name__}")
        for k, v in self.predictor_params.items():
            print_(f" --> {k}: {v}")
        print_("  Text Encoder")
        for k, v in self.text_encoder_params.items():
            print_(f"    --> {k}: {v}")
        print_("  Text-To-Slot Attention")
        for k, v in self.fusion_params.items():
            print_(f"    --> {k}: {v}")
        return



class TextOCVP_CustomTF(BaseTextOCVP):
    """
    TextOCVP variant that uses a CustomTransformer for text encoding.
    """

    def _instantiate_text_encoder(self):
        """ Instanciating a Custom-Transformer as text-encoder """
        self.text_encoder = TransformerTextEncoder(
                input_dim=self.text_encoder_params.get("input_dim"),
                num_layers=self.text_encoder_params.get("num_layers"),
                num_heads=self.text_encoder_params.get("num_heads"),
                output_dim=self.token_dim,
                vocab_size=self.text_encoder_params.get("vocab_size")
            )
        return    



class TextOCVP_T5(BaseTextOCVP):
    """
    TextOCVP variant that uses a pretrained T5 text-encoder.
    """

    def _instantiate_text_encoder(self):
        """ Instanciating a pretrained T5 as text-encoder """
        self.text_encoder = T5EncoderModel.from_pretrained("t5-small")
        freeze_params(self.text_encoder)
        self.t5_token_dim = 512  # default token dim. of pretrained T5
        return

