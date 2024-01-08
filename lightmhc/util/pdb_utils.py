"""Util functions to convert predicted coordinates to PDB file."""
from typing import Dict, List, Optional

import numpy as np

from lightmhc.util.constants import atom_types, residue_atoms, restype_1to3


def to_pdb(
    sequences: Dict[str, str],
    seq_len: Dict[str, int],
    numbers: Dict[str, List[int]],
    all_atoms: np.ndarray,
    chain_ids: str = "BA",
    lddt: Optional[np.ndarray] = None,
) -> str:
    """Convert predicted coordinates to PDB-formatted string.

    Code adapted from https://github.com/oxpig/ImmuneBuilder/blob/main/ImmuneBuilder/util.py

    Args:
        sequences: Padded sequences of each chain.
        seq_len: Length of unpadded sequences.
        numbers: List where each element is a dictionary with chains canonical numbering.
        all_atoms: Array with coordinates of all atoms.
        chain_ids: Ids of chains to select.
        lddt: Predicted CA-lddt. Shape = (n_aa,).

    Returns:
        String of the file to encode.
    """
    atom_index = 0
    pdb_lines = []
    record_type = "ATOM"

    full_sequence = "".join(sequences[chain] for chain in chain_ids)
    indexes = np.where(np.array(list(full_sequence)) != "-")[0]
    full_sequence = full_sequence.replace("-", "")
    all_atoms = all_atoms[indexes]

    chain_index = []
    for i, ch_id in enumerate(chain_ids):
        chain_index += [i] * seq_len[ch_id]

    chain_id = chain_ids[0]
    amino_number = 0
    for i, amino in enumerate(full_sequence):
        for atom in atom_types:
            if atom in residue_atoms[amino]:
                j = residue_atoms[amino].index(atom)
                pos = all_atoms[i, j]
                name = f" {atom}"
                alt_loc = ""
                res_name_3 = restype_1to3[amino]
                if chain_id != chain_ids[chain_index[i]]:
                    chain_id = chain_ids[chain_index[i]]
                    amino_number = 0
                occupancy = 1.00
                b_factor = lddt[i] if lddt is not None else 0.0
                element = atom[0]
                charge = ""
                # PDB is a columnar format, every space matters here!
                atom_line = (
                    f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                    f"{res_name_3:>3} {chain_id:>1}"
                    f"{numbers[chain_id][amino_number]:>4}    "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}"
                )
                pdb_lines.append(atom_line)
                atom_index += 1
        amino_number += 1

    return "\n".join(pdb_lines)
