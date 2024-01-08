"""Modules for dataset definition."""
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset


class StructureDataset(Dataset):
    """Generic dataset class."""

    def __init__(
        self,
        chain_lengths: Dict[str, int],
        path: Path,
        **kwargs: Any,
    ) -> None:
        """Initialize Structure Dataset.

        Needs maximum sequence length for padding, path to load files and to specify pdb/csv.

        chains: List of chains names.
        pdb_id: List of pdb ids.
        sequences: List where each element is a dictionary with each chain sequence.
        numbers: List where each element is a dictionary with canonical numbering (e.g. IMGT).
        coordinates: Dummy tensor with null coordinates of alpha and beta chains.

        Args:
            chain_lengths: Maximum length of each chain.
            path: Path to csv file containing sequences or to directory with PDB files.
        """
        self.chains: List[str]
        self.sequences_len: np.ndarray
        self.pdb_id: np.ndarray
        self.sequences: np.ndarray
        self.numbers: np.ndarray
        self.coordinates: torch.Tensor

        self.chain_lengths = chain_lengths
        self.total_length = sum([chain_lengths[chain] for chain in self.chains])

        self.load_csv_file(path)

        self._padding()
        super().__init__(**kwargs)

    def __getitem__(self, item: int) -> Data:
        """Load the data corresponding to the given index.

        The returned data object contains pdb id, alpha and beta sequences, atom coordinates.

        Args:
            item: Sample index, value will be 0 to self.len()-1.

        Returns:
            Loaded data.
        """
        data = Data(
            pdb_id=self.pdb_id[item],
            sequences=self.sequences[item],
            sequences_len=self.sequences_len[item],
            numbers=self.numbers[item],
            coordinates=self.coordinates[item],
            chain_lengths=self.chain_lengths,
            total_length=self.total_length,
        )
        if hasattr(self, "transform") and self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.pdb_id)

    def _padding(self) -> None:
        """Pad chain sequences with padding residues at the end."""
        self.sequences_len = np.array(
            [{chain: len(seq[chain]) for chain in self.chains} for seq in self.sequences]
        )
        self.sequences = np.array(
            [
                {
                    chain: seq[chain][: self.chain_lengths[chain]]
                    + "-" * (self.chain_lengths[chain] - len(seq[chain]))
                    for chain in self.chains
                }
                for seq in self.sequences
            ]
        )

    @abstractmethod
    def load_csv_file(self, file_path: Path) -> None:
        """Abstract method for csv file loading.

        Args:
            file_path: Path of the csv file.
        """


class PMHCDataset(StructureDataset):
    """pMHC Dataset class.

    Contains MHC alpha chain and peptide sequences, pdb ids and atom coordinates.
    Can select MHC only.
    """

    def __init__(
        self, chain_lengths: Dict[str, int], path: Path, **kwargs: Any
    ) -> None:
        """Initialize PMHCDataset.

        Needs maximum sequence length for padding, path to load csv.

        pdb_id: List of pdb ids.
        sequences: List where each element is a dictionary with alpha sequence.
        coordinates: Dummy tensor with null coordinates of alpha chain.

        Args:
            chain_lengths: Maximum MHC alpha and peptide sequences lengths.
            path: Path to csv file containing sequences or to directory with PDB files.
        """
        self.chains = ["A", "C"]
        super().__init__(chain_lengths, path, **kwargs)

    @staticmethod
    def concat_sequences(sequences: Union[np.ndarray, List[Dict[str, str]]]) -> List[str]:
        """Concatenate MHC alpha and peptide sequences.

        Returns:
            Concatenated MHC alpha and peptide sequences.
        """
        return [seq["A"] + seq["C"] for seq in sequences]

    def load_csv_file(self, file_path: Path) -> None:
        """Load pdb ids, sequences from a csv file.

        Args:
            file_path: Path to csv file.
        """
        df = pd.read_csv(file_path, header=0)
        self.pdb_id = df["pdb_id"].to_numpy()

        sequences_a = df["mhc"].apply(lambda s: s[:self.chain_lengths['A']])
        sequences_c = df["peptide"].apply(lambda s: s[:self.chain_lengths['C']])

        numbers_a = df["mhc"].apply(
            lambda s: list(range(1, len(s) + 1)) + (self.chain_lengths["A"] - len(s)) * [-1]
        )
        numbers_c = df["peptide"].apply(
            lambda s: list(range(1, len(s) + 1)) + (self.chain_lengths["C"] - len(s)) * [-1]
        )

        self.sequences = np.array(
            [{"A": seq_a, "C": seq_c} for (seq_a, seq_c) in zip(sequences_a, sequences_c)]
        )
        self.numbers = np.array(
            [{"A": num_a, "C": num_c} for (num_a, num_c) in zip(numbers_a, numbers_c)]
        )

        self.coordinates = torch.zeros(len(self.sequences), self.total_length, 14, 3)
