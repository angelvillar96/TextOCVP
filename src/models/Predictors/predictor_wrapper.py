"""
Implementation of a predictor wrapper module that autoregressively applies any predictor
module on a sequence of data.
"""

import torch
import torch.nn as nn
from lib.logger import print_


__all__ = [
        "PredictorWrapper"
    ]



class PredictorWrapper(nn.Module):
    """
    Wrapper module that autoregressively applies a transformer-based
    predictor module on a sequence of data.

    Args:
    -----
    exp_params: dict
        Dictionary containing the experiment parameters
    predictor: nn.Module
        Instanciated predictor module to wrap.
    """

    def __init__(self, exp_params, predictor):
        """
        Module initializer
        """
        super().__init__()
        self.exp_params = exp_params
        self.predictor = predictor
        self.predictor_name = exp_params["predictor"]["predictor_name"]
        self.predictor_params = exp_params["predictor"]["predictor_params"]

        # prediction training and inference parameters
        prediction_params = exp_params["prediction_params"]
        self.num_context = prediction_params["num_context"]
        self.num_preds = prediction_params["num_preds"]
        self.teacher_force = prediction_params["teacher_force"]
        self.input_buffer_size = prediction_params["input_buffer_size"]
        self._set_buffer_size()
        return


    def forward(self, slot_history, num_preds=None, **kwargs):
        """
        Forward pass through transformer-based prediction module.
        The model autoregressively predicts the next 'NumPreds' time-steps
        conditioned on a textual caption and the previous 'num_context' frames.

        Args:
        -----
        slot_history: torch Tensor
            Decomposed slots form the seed and predicted images.
            Shape is (B, num_frames, num_slots, slot_dim)

        Returns:
        --------
        pred_slots: torch Tensor
            Predicted subsequent slots. Shape is (B, num_preds, num_slots, slot_dim)
        """
        self._is_teacher_force()
        num_preds = num_preds if num_preds is not None else self.num_preds
        
        # Encode text caption if provided.
        text_embeddings = self.encode_text_caption(**kwargs)

        # autoregressively predict the next 'NumPreds' time-steps conditioned on caption
        predictor_input = slot_history[:, :self.num_context].clone()
        pred_slots = []
        for t in range(num_preds):
            cur_pred = self.predictor(
                    slots=predictor_input,
                    time_step=t,
                    text_embeddings=text_embeddings
                )
            next_input = slot_history[:, self.num_context+t] if self.teacher_force else cur_pred
            predictor_input = torch.cat([predictor_input, next_input.unsqueeze(1)], dim=1)
            predictor_input = self._update_buffer_size(predictor_input)
            pred_slots.append(cur_pred)
        pred_slots = torch.stack(pred_slots, dim=1)  # (B, num_preds, num_slots, slot_dim)
        return pred_slots


    def encode_text_caption(self, **kwargs):
        """ 
        Processing text-caption. This function gets the text-caption parameters from
        **kwargs, encodes the text using the corresponding text-encoder, and returns
        the encoded text-embeddings.
        """
        caption = kwargs.get("caption_tokens", None)
        if caption is None:
            raise KeyError(f"'caption_tokens' must be provided for the text-encoder.")
        device = next(self.parameters()).device
    
        # T5 Text Encoder
        if "T5" in self.predictor_name:
            attention_mask = kwargs.get("attn_masks", None)
            if attention_mask is None:
                raise KeyError(f"'attn_masks' must be provided for T5 Predictor")
            out_t5 = self.predictor.text_encoder(
                    input_ids=caption.to(device), 
                    attention_mask=attention_mask.to(device), 
                    return_dict=True
                )
            text_embeddings = out_t5.last_hidden_state
            if self.predictor.token_dim != self.predictor.t5_token_dim:
                text_embeddings = self.predictor.mlp_map_to_token_dim(text_embeddings)
        # Custom Transformer text encoder
        elif "CustomTF" in self.predictor_name:
            caption_lengths = kwargs.get("caption_lengths", None)
            if caption_lengths is None:
                raise KeyError(f"'caption_lengths' must be provided for CustomTF Pred.")
            text_embeddings = self.predictor.text_encoder(
                    text=caption.to(device), 
                    text_length=caption_lengths.to(device)
                )
        # non-condiotioned text encoder
        else:
            text_embeddings = None

        return text_embeddings        


    def _is_teacher_force(self):
        """
        Updating the teacher force value, depending on the training stage
            - In eval-mode, then teacher-forcing is always false
            - In train-mode, then teacher-forcing depends on the predictor parameters
        """
        if self.predictor.train is False:
            self.teacher_force = False
        else:
            self.teacher_force = self.exp_params["prediction_params"]["teacher_force"]
        return


    def _update_buffer_size(self, inputs):
        """
        Updating the inputs of a transformer model given the 'buffer_size'.
        We keep a moving window over the input tokens, dropping the oldest slots if the buffer
        size is exceeded.
        """
        num_inputs = inputs.shape[1]
        if num_inputs > self.input_buffer_size:
            extra_inputs = num_inputs - self.input_buffer_size
            inputs = inputs[:, extra_inputs:]
        return inputs
    
    
    def _set_buffer_size(self):
        """
        Setting the buffer size given the predicton parameters
        """
        if self.input_buffer_size is None:
            print_(f"  --> {self.predictor_name} buffer size is 'None'...")
            print_(f"       Setting it as {self.num_context = }")
            self.input_buffer_size = self.num_context
        if self.input_buffer_size < self.num_context:
            print_(f"  --> {self.predictor_name}'s {self.input_buffer_size = } is too small.")
            print_(f"  --> Using {self.num_context = } instead...")
        else:
            print_(f"  --> Using buffer size {self.input_buffer_size}...")
        return    
    
