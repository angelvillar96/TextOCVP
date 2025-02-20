"""
Methods for loading specific datasets, fitting data loaders and other
data utils functionalities
"""

from torch.utils.data import DataLoader
from CONFIG import CONFIG
from configs import get_available_configs



def load_data(exp_params, split="train"):
    """
    Loading a dataset given the parameters

    Args:
    -----
    dataset_name: string
        name of the dataset to load
    split: string
        Split from the dataset to obtain (e.g., 'train' or 'test')

    Returns:
    --------
    dataset: torch dataset
        Dataset loaded given specifications from exp_params
    """
    db_params = exp_params["dataset"]
    db_name = db_params["dataset_name"]
    DATASETS = get_available_configs("datasets")    
    if db_name not in DATASETS:
        raise NotImplementedError(
                f"""ERROR! Dataset'{db_name}' is not available.
                Please use one of the following: {DATASETS}..."""
            )    
    if db_name == "CATER_Easy":
        from data.Cater import CATER
        dataset = CATER(split=split, mode="easy", **db_params)
    elif db_name == "CATER_Hard":
        from data.Cater import CATER
        dataset = CATER(split=split, mode="hard", **db_params)
    elif db_name == "CLIPort":
        from data.CLIPort import CLIPort
        dataset = CLIPort(split=split, **db_params)
    else:
        raise NotImplementedError(
                f"""ERROR! Dataset'{db_name}' is not available.
                Please use one of the following: {DATASETS}..."""
            )
    return dataset


def build_data_loader(dataset, batch_size=8, shuffle=False):
    """
    Fitting a data loader for the given dataset

    Args:
    -----
    dataset: torch dataset
        Dataset (or dataset split) to fit to the DataLoader
    batch_size: integer
        number of elements per mini-batch
    shuffle: boolean
        If True, mini-batches are sampled randomly from the database
    """
    collate_fn = None if not hasattr(dataset, "collate_fn") else dataset.collate_fn
    data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=CONFIG["num_workers"],
            collate_fn=collate_fn
        )
    return data_loader


def unwrap_batch_data(exp_params, batch_data):
    """
    Unwrapping the batch data depending on the dataset that we are training on
    """
    others = {}
    if exp_params["dataset"]["dataset_name"] in [
                "CATER_Easy", "CATER_Hard",
                "CLIPort"
            ]:
        videos, caption_info = batch_data
        others = {**others, **caption_info}
    else:
        dataset_name = exp_params["dataset"]["dataset_name"]
        raise NotImplementedError(f"Dataset {dataset_name} is not supported...")
    return videos, others


