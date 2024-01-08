"""Modules for defining functions used for training and scoring."""
import inspect
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from torchvision.transforms import transforms

from lightmhc.data.dataset import PMHCDataset, StructureDataset
from lightmhc.data.transforms import AddNodeClass, AddPositions, CastData, ToGraph
from lightmhc.model.util import model_class_factory


def create_dataset(
    path: Path, chain_lengths: Dict[str, int], data_type: str = "pmhc"
) -> StructureDataset:
    """Creates a dataset for TCR or pMHC structure prediction.

    Args:
        path: Path to csv file containing sequences or to directory with PDB files.
        chain_lengths: Maximum chains lengths.
        data_type: Indicates the type of dataset used. Default 'pmhc'.

    Returns:
        dataset: Structure dataset.
    """
    if data_type == "pmhc":
        chain_lengths = {"A": chain_lengths["A"], "C": chain_lengths["C"]}
        return PMHCDataset(chain_lengths, path)
    raise ValueError("data_type must be pmhc.")


def get_transforms(config: DictConfig) -> transforms.Compose:
    """Returns a sequential composition of the necessary transformations for specified model.

    Args:
        config: The DictConfig used to create the model.

    Returns:
        transformations: A sequential composition of necessary transforms for the dataloader such
        that the loaded data matches the specified model.
    """
    # Convert template directory to pathlib object
    default_template_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent / "templates"
    # All models require the data casting.
    transformations: List[object] = [CastData(), AddNodeClass()]
    # All non-Transformer models require the graph to be built.
    compute_dist_matrix = config.model.compute_dist_matrix
    transformations.append(
        ToGraph(
            distance_threshold=config.data.threshold,
            coord_index=config.model.coord_index,
            add_distances=False,
            compute_dist_matrix=compute_dist_matrix,
            default_template_dir=default_template_dir,
            use_default_template=True,
        )
    )

    if config.model.pos_encoding_type != "":
        # Add node positions when positional encodings are used.
        transformations.append(AddPositions())
    return transforms.Compose(transformations)


def get_dataloader(
    dataset: StructureDataset, split_name: str, config: DictConfig, drop_last: bool = False
) -> DataLoader:
    """Create the data loader and dataframe for the given split.

    Args:
        dataset: Dataset used to create the dataloader.
        split_name: Name of the data split, must be one of "train", "val", "test".
        config: The train or score script Hydra DictConfig.
        drop_last: Whether to drop the last non-full batch when iterating over
        the dataloader (defaults to False).

    Returns:
        data_loader: data loader for the given split
    """
    should_drop_last = (split_name == "train") & drop_last

    # Create the dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=config.model.batch_size,
        shuffle=should_drop_last,
        pin_memory=True,
        drop_last=should_drop_last,
    )

    return data_loader


def get_model(config: DictConfig) -> torch.nn.Module:
    """Create Graph Neural Network according to the provided arguments.

    Args:
        config: Arguments used to create the model. It should contain the
        model class name and the different parameters used in the init function.

    Returns:
        Created object of the selected model class.
    """
    model_cls = model_class_factory(config.model_name)
    init_params_names = list(inspect.signature(model_cls.__init__).parameters)
    for parent in model_cls.__bases__:
        init_params_names += list(inspect.signature(parent.__init__).parameters)
    init_params = {
        init_param_name: getattr(config, init_param_name)
        for init_param_name in init_params_names
        if init_param_name != "self" and init_param_name in config
    }
    return model_cls(**init_params)


def load_config(
    args: DictConfig,
) -> Tuple[int, DictConfig, DictConfig]:
    """Return config values.

    Returns:
        args.seed: Seed to be used.
        args.data: Dictionary with embedding functions' config.
        args.model: Dictionary with train/score functions' config.
    """
    return args.seed.seed, args.data, args.model
