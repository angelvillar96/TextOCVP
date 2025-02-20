"""
Transition models used to provide a slot initialization at the subsequent time step
"""

import torch.nn as nn

from lib.logger import print_
from models.Blocks.attention import TransformerBlock



def get_transition_module(model_name, **kwargs):
    f"""
    Fetching the transition module to use to provide the initial slot state
    at the subsequent time step.

    Args:
    -----
    model: string
        Type of transition module to use.
    """
    
    slot_dim = kwargs.pop('slot_dim')
    if model_name in [None, ""]:
        transitor = nn.Identity()
    elif model_name == "TransformerBlock":
        transitor = TransformerBlock(
                embed_dim=slot_dim,
                pre_norm=False,
                **kwargs
            )
    else:
        raise ValueError(f"UPSI, {model_name = } was not a recognized transition module...")

    print_("Transition Module:")
    print_(f"  --> model-name: {model_name}")
    for k, v in kwargs.items():
        print_(f"  --> {k}: {v}")
    return transitor

