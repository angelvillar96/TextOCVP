"""
Computation of different metrics
"""

import os
import json
import piqa
import torch
from lib.logger import print_
from lib.utils import create_directory
from lib.visualizations import visualize_metric



class MetricTracker:
    """
    Class for computing several evaluation metrics

    Args:
    -----
    exp_path: string
        Path to the experiment directory. Needed only if use_tboard is True
    metrics: list
        List containing the metrics to evaluate
    """

    def __init__(self, exp_path=None, metrics=["accuracy"]):
        """
        Module initializer
        """
        if not isinstance(metrics, list):
            raise TypeError(f"'metrics' must be a list, not {type(metrics)}")
        valid_metrics = list(METRICS_DICT.keys())
        for metric in metrics:
            if metric not in valid_metrics:
                raise NameError(
                        f"Unknown {metric = }. Use one of {valid_metrics}"
                    )
        self.exp_path = exp_path

        self.metric_computers = {}
        print_("Using evaluation metrics:")
        for metric in metrics:
            print_(f"  --> {metric}")
            if metric not in METRICS_DICT.keys():
                raise NotImplementedError(
                        f"Unknown {metric = }. Use one of {valid_metrics}"
                    )
            self.metric_computers[metric] = METRICS_DICT[metric]()
        self.reset_results()
        return

    def reset_results(self):
        """ Reseting results and metric computers """
        self.results = {m: None for m in self.metric_computers.keys()}
        for m in self.metric_computers.values():
            m.reset()
        return

    def accumulate(self, preds, targets):
        """ Computing the different metrics and adding them to the results list """
        for _, metric_computer in self.metric_computers.items():
            _ = metric_computer.accumulate(preds=preds, targets=targets)
        return

    def aggregate(self):
        """ Aggregating the results for each metric """
        for metric, metric_computer in self.metric_computers.items():
            metric_data = metric_computer.aggregate()
            if isinstance(metric_data, (list, tuple)) and len(metric_data) == 2:
                mean_results, framewise_results = metric_data
                self.results[metric] = {}
                self.results[metric]["mean"] = mean_results
                self.results[metric]["framewise"] = framewise_results
            else:
                self.results[metric] = metric_data
        return

    def get_results(self):
        """ Retrieving results """
        return self.results

    def summary(self):
        """ Printing and fetching the results """
        print_("RESULTS:")
        print_("--------")
        for metric in self.metric_computers.keys():
            if isinstance(self.results[metric], dict):
                result = round(self.results[metric]["mean"], 3)
            else:
                result = round(self.results[metric].item(), 3)
            print_(f"  {metric}:  {result}")
        return self.results

    def save_results(self, exp_path, fname):
        """ Storing results into JSON file """
        results_dir = os.path.join(exp_path, "results", fname)
        create_directory(dir_path=results_dir)
        results_file = os.path.join(results_dir, "results.json")

        # converting to list/float and rounding numerical values
        cur_results = {}
        for metric in self.results:
            if self.results[metric] is None:
                continue
            elif isinstance(self.results[metric], dict):
                cur_results[metric] = {}
                cur_results[metric]["mean"] = round(self.results[metric]["mean"], 5)
                cur_results[metric]["framewise"] = []
                for r in self.results[metric]["framewise"].cpu().detach().tolist():
                    cur_results[metric]["framewise"].append(round(r, 5))
            else:
                cur_results[metric] = round(self.results[metric].item(), 5)

        # loading already saved values and overwriting some if necessary.
        if os.path.exists(results_file):
            with open(results_file) as file:
                old_results = json.load(file)
            for k, v in old_results.items():
                if k not in cur_results.keys():
                    cur_results[k] = v

        # saving current results
        with open(results_file, "w") as file:
            json.dump(cur_results, file)
        return

    def make_plots(self, start_idx=5, savepath=None, prefix="_", **kwargs):
        """ Making and saving plots for each of the framewise results"""
        for metric in self.results:
            if not isinstance(self.results[metric], dict):
                continue
            cur_vals = [
                round(r, 5) for r in self.results[metric]["framewise"].cpu().detach().tolist()
            ]
            cur_savepath = os.path.join(savepath, f"results_{metric}.png")
            visualize_metric(
                    vals=cur_vals,
                    start_x=start_idx,
                    title=metric,
                    savepath=cur_savepath,
                    xlabel="Frame"
                )
        return



class Metric:
    """
    Base class for metrics
    """

    def __init__(self):
        """ Metric initializer """
        self.results = None
        self.reset()

    def reset(self):
        """ Reseting precomputed metric """
        raise NotImplementedError("BaseMetric does not implement 'reset' function")

    def accumulate(self):
        """ """
        raise NotImplementedError("BaseMetric does not implement 'accumulate' function")

    def aggregate(self):
        """ """
        raise NotImplementedError("BaseMetric does not implement 'aggregate' function")

    def _shape_check(self, tensor, name="Preds"):
        """ """
        if len(tensor.shape) not in [3, 4, 5]:
            raise ValueError(
                f"{name} with {tensor.shape = }, but it must have one of the folling:\n"
                f" - (B, F, C, H, W) for frame or heatmap prediction.\n"
                f" - (B, F, D) or (B, F, N_joints, N_coords) for pose skeleton prediction"
            )



class PSNR(Metric):
    """ Peak Signal-to-Noise ratio computer """

    LOWER_BETTER = False

    def __init__(self):
        """ """
        super().__init__()
        self.values = []

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_psnr = piqa.psnr.psnr(preds, targets)
        cur_psnr = cur_psnr.view(B, F)
        self.values.append(cur_psnr)
        return cur_psnr.mean()

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values



class SSIM(Metric):
    """ Structural Similarity computer """

    LOWER_BETTER = False

    def __init__(self, window_size=11, sigma=1.5, n_channels=3):
        """ """
        self.ssim = piqa.ssim.SSIM(
                window_size=window_size,
                sigma=sigma,
                n_channels=n_channels,
                reduction=None
            )
        super().__init__()
        self.values = []

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if self.ssim.kernel.device != preds.device:
            self.ssim = self.ssim.to(preds.device)

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_ssim = self.ssim(preds, targets)
        cur_ssim = cur_ssim.view(B, F)
        self.values.append(cur_ssim)
        return cur_ssim.mean()

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values



class LPIPS(Metric):
    """ Learned Perceptual Image Patch Similarity computers """

    LOWER_BETTER = True

    def __init__(self, network="alex", pretrained=True, reduction=None):
        """ """
        self.lpips = piqa.lpips.LPIPS(
                network=network,
                pretrained=pretrained,
                reduction=reduction
            )
        super().__init__()
        self.values = []

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if not hasattr(self.lpips, "device"):
            self.lpips = self.lpips.to(preds.device)
            self.lpips.device = preds.device

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_lpips = self.lpips(preds, targets)
        cur_lpips = cur_lpips.view(B, F)
        self.values.append(cur_lpips)
        return cur_lpips.mean()

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values



METRICS_DICT = {
        "psnr": PSNR,
        "ssim": SSIM,
        "lpips": LPIPS
    }
