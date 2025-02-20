"""
Loss functions and loss-related utils
"""

import torch
import torch.nn as nn

from lib.logger import log_info, print_



class LossTracker:
    """
    Class for computing, weighting and tracking several loss functions

    Args:
    -----
    loss_params: dict
        Loss section of the experiment paramteres JSON file
    """

    def __init__(self, loss_params, **kwargs):
        """
        Loss tracker initializer
        """
        if not isinstance(loss_params, list):
            raise TypeError(f"Loss_params must be a list, not {type(loss_params)}")
        LOSSES = list(LOSS_DICT.keys())
        for loss in loss_params:
            if loss["type"] not in LOSSES:
                raise NotImplementedError(
                        f"Loss {loss['type']} not implemented. Use one of {LOSSES}"
                    )

        self.device = kwargs.get("device", 'cpu')
        self.loss_computers = {}
        for loss in loss_params:
            loss['device'] = self.device
            loss_type, loss_weight = loss["type"], loss["weight"]
            self.loss_computers[loss_type] = {}
            self.loss_computers[loss_type]["metric"] = get_loss(loss_type, **loss)
            self.loss_computers[loss_type]["weight"] = loss_weight
        self.reset()
        return

    def reset(self):
        """
        Reseting loss tracker
        """
        self.loss_values = {loss: [] for loss in self.loss_computers.keys()}
        self.loss_values["_total"] = []
        return

    def __call__(self, **kwargs):
        """
        Wrapper for calling accumulate
        """
        self.accumulate(**kwargs)

    def accumulate(self, **kwargs):
        """
        Computing the different metrics, weigting them according to their multiplier,
        and adding them to the results list.
        """
        total_loss = 0
        for loss in self.loss_computers:
            loss_val = self.loss_computers[loss]["metric"](**kwargs)
            self.loss_values[loss].append(loss_val)
            total_loss = total_loss + loss_val * self.loss_computers[loss]["weight"]
        self.loss_values["_total"].append(total_loss)
        return

    def aggregate(self):
        """
        Aggregating the results for each metric
        """
        self.loss_values["mean_loss"] = {}
        for loss in self.loss_computers:
            self.loss_values["mean_loss"][loss] = torch.stack(self.loss_values[loss]).mean()
        self.loss_values["mean_loss"]["_total"] = torch.stack(self.loss_values["_total"]).mean()
        return

    def get_last_losses(self, total_only=False):
        """
        Fetching the last computed loss value for each loss function
        """
        if total_only:
            last_losses = self.loss_values["_total"][-1]
        else:
            last_losses = {loss: loss_vals[-1] for loss, loss_vals in self.loss_values.items()}
        return last_losses

    def summary(self, log=True, get_results=True):
        """
        Printing and fetching the results
        """
        if log:
            log_info("LOSS VALUES:")
            log_info("--------")
            for loss, loss_value in self.loss_values["mean_loss"].items():
                log_info(f"  {loss}:  {round(loss_value.item(), 5)}")

        return_val = self.loss_values["mean_loss"] if get_results else None
        return return_val



def get_loss(loss_type="mse", **kwargs):
    """
    Loading a function of object for computing a loss
    """
    LOSSES = list(LOSS_DICT.keys())
    if loss_type not in LOSSES:
        raise NotImplementedError(f"{loss_type = } not available. Use one of {LOSSES}")

    print_(f"Creating loss function of type: {loss_type}")
    loss = LOSS_DICT[loss_type]()
    return loss



class Loss(nn.Module):
    """
    Base class for custom loss functions
    """

    REQUIRED_ARGS = []

    def __init__(self):
        super().__init__()

    def _unpack_kwargs(self, **kwargs):
        """
        Fetching the required arguments from kwargs
        """
        out = []
        for arg in self.REQUIRED_ARGS:
            if arg not in kwargs:
                cls_name = self.__class__.__name__
                raise ValueError(
                        f"Required '{arg = }' not in {kwargs.keys() = } in {cls_name}"
                    )
            out.append(kwargs[arg])
        if len(out) == 1:
            out = out[0]
        return out



class MSELoss(Loss):
    """
    Overriding MSE Loss
    """
    
    REQUIRED_ARGS = ["pred_imgs", "target_imgs"]

    def __init__(self):
        """
        Module initializer
        """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, **kwargs):
        """
        Computing loss
        """
        preds, targets = self._unpack_kwargs(**kwargs)
        loss = self.mse(preds, targets)
        return loss



class PredImgMSELoss(MSELoss):
    """
    Pretty much the same MSE Loss.
    Use this loss on predicted images, while still enforcing
    MSELoss on predicted slots
    """

    REQUIRED_ARGS = ["pred_imgs", "target_imgs"]



class PredSlotMSELoss(MSELoss):
    """
    MSE Loss used on slot-like representations.
    This can be used when forecasting future slots.
    """

    REQUIRED_ARGS = ["pred_slots", "target_slots"]



class PredFeatsMSELoss(MSELoss):
    """
    MSE Loss used on patch-feature representations.
    This can be used when training DINOSAUR-based models
    """

    REQUIRED_ARGS = ["preds_feats", "targets_feats"]




# LOSS Dictionary
LOSS_DICT = {
    "mse": MSELoss,
    "pred_img_mse": PredImgMSELoss,
    "pred_slot_mse": PredSlotMSELoss,
    "pred_feature_mse": PredFeatsMSELoss
}