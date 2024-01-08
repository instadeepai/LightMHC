"""Handles conversion of predicted tensors to PDB files."""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.protocols.idealize import IdealizeMover

from lightmhc.util.pdb_utils import to_pdb


class StructurePrediction:
    """Class that handles predictions of the model.

    Given the sequences and predicted coordinates, save the structure.
    """

    def __init__(
        self,
        pdb_id: str,
        sequences: Dict[str, str],
        sequences_len: Dict[str, int],
        numbers: Dict[str, List[int]],
        predictions: torch.Tensor,
        chain_ids: List[str],
        lddt: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize class.

        Need sequences of each chain and predicted coordinates.

        Args:
            pdb_id: Identifier of the predicted structure.
            sequences: Dictionary containing padded sequences of the structure.
            sequences_len: Dictionary containing sequences lengths before padding.
            numbers: Dictionary with chains numbering.
            predictions: Predicted atom coordinates and node embedding of the ensemble.
                        Shape = (n_models, n_aa, 14, 3).
            chain_ids: List of chains ids in the predicted structure.
            lddt: Predicted CA-lddt. Shape = (n_aa,).
        """
        self.pdb_id = pdb_id
        self.sequences = sequences
        self.sequences_len = sequences_len
        self.numbers = numbers
        self.predictions = predictions
        self.lddt = lddt
        self.n_models = predictions.shape[0]

        self.chain_ids = "".join(chain_ids)

    def save(self, file_path: Path) -> None:
        """Save structure from a single model.

        The predicted coordinates are converted to a PDB that is saved.

        Args:
            file_path: Path of the file.
        """
        atoms = self.predictions
        file_path.parent.mkdir(parents=True, exist_ok=True)
        unrefined = to_pdb(
            self.sequences, self.sequences_len, self.numbers, atoms, self.chain_ids, self.lddt
        )

        with open(str(file_path), "w+") as file:
            file.write(unrefined)


def dump_pdb(
    output_dir: Path,
    chain_ids: List[str],
    preds: torch.Tensor,
    pdb_ids: np.ndarray,
    sequences: np.ndarray,
    sequences_len: np.ndarray,
    numbers: np.ndarray,
) -> None:
    """From predicted coordinates and associated sequences, create and save a PDB file.

    This function iterates over all the samples contained in a tensor of coordinates.

    Args:
        output_dir: Output directory where PDB file is saved.
        chain_ids: List of chains ids in the predicted structure.
        preds: Predicted atom coordinates. Shape = (n_samples, n_aa, 14, 3).
        pdb_ids: Identifiers of the predicted structures.
        sequences: Dictionary containing padded sequences of the structures.
        sequences_len: Dictionary containing sequences lengths before padding.
        numbers: Dictionary with chains numbering.
    """
    for i in range(preds.shape[0]):
        prediction = StructurePrediction(
            pdb_ids[i],
            sequences[i],
            sequences_len[i],
            numbers[i],
            preds[i],
            chain_ids,
        )
        prediction.save(file_path=output_dir / f"{pdb_ids[i]}.pdb")


def fix_structures(input_folder: Path, output_folder: Path) -> None:
    """Fix structures using Rosetta IdealizeMover.

    The mover alters atom positions to ensure bond lengths and angles are consistent.

    Args:
        input_folder: Folder containing the structures to be fixed.
        output_folder: Folder containing the fixed structures.
    """
    # Initialize pyrosetta and IdealizeMover.
    init("-out:level 0")
    min_mover = IdealizeMover()
    min_mover.fast(True)

    output_folder.mkdir(parents=True, exist_ok=True)

    for file in input_folder.iterdir():
        out_file = output_folder / file.name
        pose = pose_from_pdb(str(file))
        min_mover.apply(pose)
        pose.dump_pdb(str(out_file))
