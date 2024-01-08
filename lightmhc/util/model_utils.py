"""Modules utilities for GNN structure models."""
from __future__ import annotations

from enum import Enum

import torch
from torch_geometric.data import Data

from lightmhc.data.dataset import PMHCDataset
from lightmhc.rigids import (
    Rigid,
    all_atoms_from_global_reference_frames,
    global_frames_from_bb_frame_and_torsion_angles,
    rigid_from_tensor,
)
from lightmhc.util.constants import all_frames_atoms, r2n


class OutDim(Enum):
    """Standard output dimension of TrEGNN block for each output type.

    The following standard keys are defined:
    * `BACKBONE`: Backbone + CB atoms: 5*3=15.
    * `TORSION`: Torsion angles sine, cosine: 6*2 = 12.
    """

    BACKBONE = 15
    TORSION = 12


def get_bb_frames(bb_coords: torch.Tensor, sequences: list[str]) -> Rigid:
    """Convert backbone coordinates into a backbone frame.

    Frames stemming from NaN coordinates are zeroed to avoid nan loss issues.
    Frames are extracted using N, CA and C coordinates following AF2 process.

    Args:
        bb_coords: Predicted backbone coordinates. Shape = (batch_size, n_aa, 4, 3).
        sequences: Sequences of the batch. Shape = (batch_size, n_aa).

    Returns:
        Predicted backbone frames. Shape = (batch_size, n_aa).

    """
    batch_size = bb_coords.shape[0]
    device = bb_coords.device

    # Get atom indexes to compute frames of ground truth.
    idx_atoms_frames = [[all_frames_atoms[res][0] for res in sequence] for sequence in sequences]
    idx_atoms_frames_torch = torch.tensor(
        idx_atoms_frames, device=device
    )  # Shape: (batch, n_aa, 3).
    idx_atoms_frames_torch = idx_atoms_frames_torch.unsqueeze(-2)  # Shape: (batch, n_aa, 1, 3).
    coords_frames = torch.gather(
        bb_coords.unsqueeze(-1).repeat(1, 1, 1, 1, 3),
        2,
        idx_atoms_frames_torch.unsqueeze(-2).repeat(1, 1, 1, 3, 1),
    ).transpose(
        -1, -2
    )  # Shape: (batch, n_aa, 1, 3 = n_bb_atoms_in_frame, 3 = xyz).
    bb_frames = rigid_from_tensor(coords_frames)
    bb_frames.rot.tensor[bb_frames.rot.tensor.isnan()] = 0.0
    # Reshape frames to (batch_size, max_frames * seq_len).
    bb_frames = bb_frames.view((batch_size, -1))  # Shape = (batch, n_aa).
    return bb_frames


def convert_to_coord(
    rigid_in: Rigid, torsions: torch.Tensor, sequences: list[str]
) -> tuple[torch.Tensor, Rigid]:
    """Convert predicted torsion angles and local frames to atom coordinates and global frames.

    Args:
        rigid_in: Predicted backbone frames. Shape = (batch, n_aa).
        torsions: Predicted torsion angles. Shape = (batch, n_aa, 5, 2).
        sequences: List of sequences in a given batch.

    Returns:
        all_atoms: Predicted coordinates of all atoms in a batch.
                        Shape = (batch, n_aa, 14, 3).
        all_reference_frames: Predicted global frames on the batch. Shape = (batch, n_aa, 6).
    """
    all_reference_frames = global_frames_from_bb_frame_and_torsion_angles(
        rigid_in, torsions, sequences
    )
    all_atoms = all_atoms_from_global_reference_frames(all_reference_frames, sequences)

    return all_atoms, all_reference_frames


def dataset_to_encodings(
    dataset: Data,
    device: torch.device,
    one_hot: bool = False,
    backward_compatibility: bool = False,
) -> tuple[torch.Tensor, str]:
    """Convert dataset into one-hot encoded embeddings and concatenated sequence.

    Args:
        dataset: Dataset to be converted.
        device: Device to store tensors.
        one_hot: One-hot encodings if True, otherwise residue index.
        backward_compatibility: If True, chain order is reversed to get correct chain encoding.

    Returns:
        encodings: One-hot encoding of alpha+beta sequences.
        full_seq: Concatenated alpha+beta sequences.
    """
    sequences = dataset.sequences
    chain_ids = list(sequences.keys())
    chain_ids.sort(reverse=False)
    with torch.no_grad():
        encodings = get_encoding(
            sequences, chain_ids, one_hot=one_hot, backward_compatibility=backward_compatibility
        )
        encodings = encodings.to(device=device)

        if chain_ids == ["A", "C"]:
            full_seq = PMHCDataset.concat_sequences([sequences])
        else:
            full_seq = sequences["A"]

    return encodings, full_seq[0]


def get_encoding(
    sequence_dict: dict[str, str],
    chain_ids: list[str],
    nb_classes: int = 22,
    one_hot: bool = False,
    backward_compatibility: bool = False,
) -> torch.Tensor:
    """Convert sequences of each chain to encoded vectors.

    One-hot mode: one-hot encoded vectors.
    If not, each AA encoding is set to its index.
    Add chain_number * num_classes to distinguish chains.

    Both amino-acid type and chain number are one-hot encoded.

    Args:
        sequence_dict: Dictionary containing the sequence of each chain.
        chain_ids: Chains to encode.
        nb_classes: Number of classes. Default 22.
        one_hot: One-hot encodings if True. Else residue index.
        backward_compatibility: If True, chain order is reversed to get correct chain encoding.

    Returns:
        Encoded matrix. Shape = (n_aa, ) or (n_aa, num_classes).
    """
    encodings = []

    for chain_number, chain in enumerate(chain_ids):
        if backward_compatibility:
            chain_number = len(chain_ids) - chain_number - 1

        seq = sequence_dict[chain]
        one_hot_amino = torch.nn.functional.one_hot(
            torch.tensor([r2n.get(x, len(r2n)) for x in seq]), nb_classes
        )
        if one_hot:
            one_hot_region = torch.nn.functional.one_hot(
                chain_number * torch.ones(len(seq), dtype=torch.int64), 2
            )
            encoding = torch.hstack([one_hot_amino, one_hot_region])
            encoding = encoding.to(torch.float32)
        else:
            encoding = torch.where(one_hot_amino == 1)[1]
            encoding += chain_number * nb_classes
            encoding = encoding.to(torch.int32)

        encodings.append(encoding)
    return torch.cat(encodings, dim=0).to(encoding.dtype)
