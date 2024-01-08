"""Test Units for model utils."""
# pylint: disable=redefined-outer-name

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from torchvision import transforms

from lightmhc.core import get_dataloader
from lightmhc.data.dataset import PMHCDataset
from lightmhc.data.transforms import AddNodeClass
from lightmhc.model.layer import FeatureBuilder


@pytest.mark.parametrize(
    ("node_emb_type"),
    [
        ("class"),
    ],
)
def test_feature_builder(
    node_emb_type: str,
    gnn_structure_fixtures: Path,
    dim_emb: int = 16,
    num_node_classes: int = 43,
) -> None:
    """Test FeatureBuilder.

    Args:
        node_emb_type: Type of node embedding.
        gnn_structure_fixtures: Path to fixtures assets.
        dim_emb: Embedding dimension.
        num_node_classes: Number of node classes. Default 43.

    Tests:
        Features obtained from the feature builder have the correct shape.
    """
    chain_lengths = {"A": 180, "C": 13}

    fixtures_path = gnn_structure_fixtures / "pmhc/dataset.csv"
    transformations = transforms.Compose([AddNodeClass()])  # get_transforms(hydra_config, "train")

    config = OmegaConf.create({"model": {"seed": 0, "num_workers": 1, "batch_size": 2}})
    dataset = PMHCDataset(
        chain_lengths=chain_lengths,
        path=fixtures_path,
        transform=transformations,
    )

    dataloader = get_dataloader(dataset, "test", config, drop_last=False)

    data = next(iter(dataloader))
    feature_builder = FeatureBuilder(
        node_emb_type,
        dim_emb,
        num_node_classes,
    )

    output = feature_builder(data)

    num_nodes = sum([chain_lengths[chain] for chain in chain_lengths])
    assert list(output.size()) == [config.model.batch_size * num_nodes, dim_emb]
