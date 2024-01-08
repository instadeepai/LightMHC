"""Tests for inference functions."""
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import DictConfig

from lightmhc.core import create_dataset, get_dataloader, get_model, get_transforms
from lightmhc.inference import inference_step, partition_csv, workflow

chain_lengths = {"A": 180, "C": 13}


def test_partition_csv(gnn_structure_fixtures: Path) -> None:
    """Test partition function.

    Args:
        gnn_structure_fixtures: Folder containing dummy csv dataset.

    Tests:
        Partitioned CSV exists and has correct shape.
    """
    input_csv_path = gnn_structure_fixtures / "pmhc/dataset.csv"
    n_cpus = 2
    partition_list = partition_csv(input_csv_path, n_cpus)
    assert len(partition_list) == 2
    for csv_file in partition_list:
        csv_file = Path(csv_file)
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        assert df.shape == (1, 3)


@pytest.mark.heavy()
def test_inference_step(gnn_structure_fixtures: Path, test_config: DictConfig) -> None:
    """Test step function from the inference loop.

    Args:
        gnn_structure_fixtures: Folder containing dummy csv dataset.
        test_config: Dummy test configuration.

    Tests:
        Predictions and labels have same number of samples.
        All positive samples have been seen (for pMTnet).
    """
    device = torch.device("cpu")

    dataset = create_dataset(
        gnn_structure_fixtures / "pmhc/dataset.csv", chain_lengths, "pmhc"
    )
    transformations = get_transforms(test_config)
    dataset.transform = transformations
    dataloader = get_dataloader(dataset, "test", test_config)

    model = get_model(test_config.model).to(device)

    for data_test in dataloader:
        preds = inference_step(model, data_test, device)
        assert preds.shape == (len(dataset), dataloader.dataset.total_length, 14, 3)


@pytest.mark.heavy()
def test_workflow(
    gnn_structure_fixtures: Path, gnn_structure_tempdir: Path, test_config: DictConfig
) -> None:
    """Test scoring loop.

    Args:
        gnn_structure_fixtures (fixture): Folder containing dummy csv dataset.
        gnn_structure_tempdir:  Directory to store saved models and parameters.
        test_config: Dummy test_config.

    Tests:
        Test metrics file correctly saved after the loop.
    """
    device = torch.device("cpu")
    input_csv_path = gnn_structure_fixtures / "pmhc/dataset.csv"
    df = pd.read_csv(input_csv_path)

    test_config.model.checkpoint_path = str(gnn_structure_fixtures / "pmhc/dummy_gnn_ckpt.pt")

    workflow(
        test_config,
        input_csv_path,
        output_dir=gnn_structure_tempdir,
        device=device,
        fix_pdb=True,
    )

    workflow_dir = gnn_structure_tempdir / "workflow_0"

    # Check workflow-specific directory is removed after pdb fixed.
    assert not workflow_dir.exists()
    pdb_files = [f for f in gnn_structure_tempdir.iterdir() if f.suffix == ".pdb"]
    assert len(pdb_files) == df.shape[0]
