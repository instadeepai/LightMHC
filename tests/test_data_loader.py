"""Test Units for data processing, dataset creation and dataloader object."""
# pylint: disable=redefined-outer-name
from pathlib import Path
from typing import Dict

import omegaconf
import pytest
from torch_geometric.utils import contains_isolated_nodes, contains_self_loops

from lightmhc.core import get_dataloader, get_transforms
from lightmhc.data.dataset import PMHCDataset


@pytest.mark.parametrize(
    ("chain_lengths"),
    [
        ({"A": 180, "C": 13}),
    ],
)
def test_to_graph_transform(
    chain_lengths: Dict[str, int],
    gnn_structure_fixtures: Path,
) -> None:
    """Tests the ToGraph transformation applied to the dataloader.

    Args:
        chain_lengths: Length of each chain.
        gnn_structure_fixtures: Path to fixtures containing PDB files to load.
    """
    fixtures_path = gnn_structure_fixtures / "pmhc/dataset.csv"
    config = omegaconf.OmegaConf.create(
        {
            "data": {"threshold": 8},
            "model": {
                "seed": 0,
                "num_workers": 1,
                "batch_size": 2,
                "use_coords": False,
                "pos_encoding_type": "learned",
                "compute_dist_matrix": True,
                "coord_index": 0,
            },
        }
    )

    transformations = get_transforms(config)

    dataset = PMHCDataset(
        chain_lengths=chain_lengths,
        path=fixtures_path,
        transform=transformations,
    )
    dataloader = get_dataloader(dataset, "test", config, drop_last=False)

    for data in dataloader:
        num_chains = len(data.chain_lengths)
        assert contains_self_loops(data.edge_index) is False
        assert contains_isolated_nodes(data.edge_index) is False
        assert len(data.edge_classes.unique()) == (num_chains**2 + num_chains)
