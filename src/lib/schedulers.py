"""
Implementation of learning rate schedulers, early stopping and other utils
for improving optimization
"""

from lib.logger import print_


def update_scheduler(scheduler, exp_params, end_epoch=False, **kwargs):
    """
    Updating the learning rate scheduler by performing the scheduler step.

    Args:
    -----
    scheduler: torch.optim
        scheduler to evaluate
    exp_params: dictionary
        dictionary containing the experiment parameters
    control_metric: float/torch Tensor
        Last computed validation metric.
        Needed for plateau scheduler
    iter: float
        number of optimization step.
        Needed for cyclic, cosine and exponential schedulers
    end_epoch: boolean
        True after finishing a validation epoch or certain number of iterations.
        Triggers schedulers such as plateau or fixed-step
    """
    scheduler_type = exp_params["training"]["scheduler"]
    if(scheduler_type == "cosine_annealing" and not end_epoch):
        scheduler.step()
    else:
        pass
    return



class IdentityScheduler:
    """ Dummy scheduler """
    def __init__(self, init_lr):
        """ """
        self.init_lr = init_lr
        pass
        
    def update_lr(self, step):
        """ """
        return self.init_lr

    def step(self, iter):
        """ Scheduler step """
        return

    def state_dict(self):
        """
        State dictionary
        """
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loading state dictinary
        """
        return
    


class LRWarmUp:
    """
    Class for performing learning rate warm-ups. We increase the learning rate
    during the first few iterations until it reaches the standard LR

    Args:
    -----
    init_lr: float
        initial learning rate
    warmup_steps: integer
        number of optimization steps to warm up for
    """

    def __init__(self, init_lr, warmup_steps):
        """
        Initializer
        """
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.active = True
        self.final_step = -1

    def __call__(self, iter, epoch, optimizer):
        """
        Computing actual learning rate and updating optimizer
        """
        if(iter > self.warmup_steps):
            if(self.active):
                self.final_step = iter
                self.active = False
                lr = self.init_lr
                print_("Finished learning rate warmup period...")
                print_(f"  --> Reached iter {iter} >= {self.warmup_steps}")
                print_(f"  --> Reached at epoch {epoch}")
        else:
            if iter >= 0:
                lr = self.init_lr * (iter / self.warmup_steps)
                for params in optimizer.param_groups:
                    params["lr"] = lr
        return

    def state_dict(self):
        """
        State dictionary
        """
        state_dict = {key: value for key, value in self.__dict__.items()}
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loading state dictinary
        """
        self.init_lr = state_dict.init_lr
        self.warmup_steps = state_dict.warmup_steps
        self.active = state_dict.active
        self.final_step = state_dict.final_step
        return



class WarmupVSScehdule:
    """
    Orquestrator module that calls the LR-Warmup module during the warmup iterations,
    and makes calls the LR Scheduler once warmup is finished.
    """

    def __init__(self, optimizer, lr_warmup, scheduler):
        """
        Initializer of the Warmup-Scheduler orquestrator
        """
        self.optimizer = optimizer
        self.lr_warmup = lr_warmup
        self.scheduler = scheduler
        return

    def __call__(self, iter, epoch, exp_params, end_epoch, control_metric=None):
        """
        Calling either LR-Warmup or LR-Scheduler
        """
        if self.lr_warmup.active:
            self.lr_warmup(iter=iter, epoch=epoch, optimizer=self.optimizer)
        else:
            update_scheduler(
                    scheduler=self.scheduler,
                    exp_params=exp_params,
                    iter=iter - self.lr_warmup.final_step - 1,
                    end_epoch=end_epoch,
                    control_metric=control_metric
            )
        return


#
