"""
Modules for the initalization of the slots on SlotAttention and SAVI
"""

import torch
import torch.nn as nn
from math import sqrt
from lib.logger import print_


def get_initializer(mode, slot_dim, num_slots):
    """
    Fetching the initializer module of the slots

    Args:
    -----
    model: string
        Type of initializer to use. Valid modes are {INITIALIZERS}
    slot_dim: int
        Dimensionality of the slots
    num_slots: int
        Number of slots to initialize
    """
    if mode == "Learned":
        intializer = Learned(slot_dim=slot_dim, num_slots=num_slots)
    elif mode == "LearnedRandom":
        intializer = LearnedRandom(slot_dim=slot_dim, num_slots=num_slots)
    else:
        raise ValueError(f"UPSI, {mode = } is not a recongnized initializer...")

    print_("Initializer:")
    print_(f"  --> mode={mode}")
    print_(f"  --> slot_dim={slot_dim}")
    print_(f"  --> num_slots={num_slots}")
    return intializer



class Learned(nn.Module):
    """
    Learned Initialization.
    Slots are randomly initialized from a Gaussian and learned by backpropagation
    """

    def __init__(self, slot_dim, num_slots):
        """ Module intializer """
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim))

        with torch.no_grad():
            limit = sqrt(6.0 / (1 + slot_dim))
            torch.nn.init.uniform_(self.slots, -limit, limit)
        self.slots.requires_grad_()
        return

    def forward(self, batch_size, **kwargs):
        """ Sampling random Gaussian slots """
        slots = self.slots.repeat(batch_size, 1, 1)
        return slots



class LearnedRandom(nn.Module):
    """
    Learned random intialization. This is the default mode used in SlotAttention.
    Slots are randomly sampled from a Gaussian distribution. However, the statistics of this
    distribution (mean vector and diagonal of covariance) are learned via backpropagation
    """

    def __init__(self, slot_dim, num_slots):
        """ Module intializer """
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

        with torch.no_grad():
            limit = sqrt(6.0 / (1 + slot_dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_sigma, -limit, limit)
        return

    def forward(self, batch_size, **kwargs):
        """
        Sampling random slots from the learned gaussian distribution
        """
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_sigma.expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=self.slots_mu.device)
        return slots

