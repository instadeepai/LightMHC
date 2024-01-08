"""Test functions for dataset class."""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import torch

from lightmhc.data.dataset import PMHCDataset


@pytest.mark.parametrize(
    ("data_type", "chain_lengths"),
    [
        ("pmhc", {"A": 180, "C": 9}),
    ],
)
def test_padding(
    data_type: str, chain_lengths: Dict[str, int], gnn_structure_fixtures: Path
) -> None:
    """Test padding function.

    Tests:
        Sequences are correctly padded with glycine residues.
    """
    fixtures_path = gnn_structure_fixtures / data_type

    dataset = PMHCDataset(
        chain_lengths=chain_lengths,
        path=fixtures_path / "dataset.csv",
    )
    df = pd.read_csv(fixtures_path / "dataset.csv")
    assert "-" not in set(dataset.sequences[0]["A"][-(chain_lengths["A"] - len(df["mhc"][0])) :])
    assert "-" not in set(
        dataset.sequences[0]["C"][-(chain_lengths["C"] - len(df["peptide"][0])) :]
    )
    assert "-" not in set(dataset.sequences[1]["A"][-(chain_lengths["A"] - len(df["mhc"][1])) :])
    assert set(dataset.sequences[1]["C"][-(chain_lengths["C"] - len(df["peptide"][1])) :]) == {"-"}


@pytest.mark.parametrize(
    ("data_type", "expected_ids", "chain_lengths"),
    [
        ("pmhc", ["1a1m", "1a1n"], {"A": 180, "C": 9}),
    ],
)
def test_load_csv(
    data_type: str,
    expected_ids: List[str],
    chain_lengths: Dict[str, int],
    gnn_structure_fixtures: Path,
) -> None:
    """Test dataset creation from csv file.

    Tests:
        Dataset attributes are correct.
    """
    fixtures_path = gnn_structure_fixtures / data_type

    dataset = PMHCDataset(
        chain_lengths=chain_lengths,
        path=fixtures_path / "dataset.csv",
    )

    assert dataset.pdb_id.tolist() == expected_ids
    assert dataset.chain_lengths == chain_lengths
    assert np.equal(dataset.sequences, torch.load(fixtures_path / "dataset_csv_sequences.pt")).all()
    assert np.equal(
        dataset.sequences_len, torch.load(fixtures_path / "dataset_csv_sequences_len.pt")
    ).all()
    assert np.equal(dataset.numbers, torch.load(fixtures_path / "dataset_csv_numbers.pt")).all()
    assert torch.equal(dataset.coordinates, torch.zeros(2, dataset.total_length, 14, 3))
