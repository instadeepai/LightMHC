"""Constants used to compute atom coordinates."""
from typing import Any, Dict, List, Tuple, Union

# Adapted from https://github.com/oxpig/ImmuneBuilder/blob/main/ImmuneBuilder/constants.py

# 1. General residue constants

# Ordered list of residue types.
restypes = "ARNDCQEGHILKMFPSTWYV-"

# Residue name conversion one to three.
restype_1to3: Dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "-": "PAD",
}

# 3 to 1 residue name conversion.
restype_3to1: Dict[str, str] = {v: k for k, v in restype_1to3.items()}

# Index of each residue type.
r2n: Dict[str, int] = {x: i for i, x in enumerate(restypes)}

# How atoms are sorted in MLAb:
residue_atoms: Dict[str, List[str]] = {
    "A": ["CA", "N", "C", "CB", "O"],
    "C": ["CA", "N", "C", "CB", "O", "SG"],
    "D": ["CA", "N", "C", "CB", "O", "CG", "OD1", "OD2"],
    "E": ["CA", "N", "C", "CB", "O", "CG", "CD", "OE1", "OE2"],
    "F": ["CA", "N", "C", "CB", "O", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "G": [
        "CA",
        "N",
        "C",
        "CA",
        "O",
    ],  # G has no CB so I am padding it with CA so the Os are aligned
    "H": ["CA", "N", "C", "CB", "O", "CG", "CD2", "CE1", "ND1", "NE2"],
    "I": ["CA", "N", "C", "CB", "O", "CG1", "CG2", "CD1"],
    "K": ["CA", "N", "C", "CB", "O", "CG", "CD", "CE", "NZ"],
    "L": ["CA", "N", "C", "CB", "O", "CG", "CD1", "CD2"],
    "M": ["CA", "N", "C", "CB", "O", "CG", "CE", "SD"],
    "N": ["CA", "N", "C", "CB", "O", "CG", "ND2", "OD1"],
    "P": ["CA", "N", "C", "CB", "O", "CG", "CD"],
    "Q": ["CA", "N", "C", "CB", "O", "CG", "CD", "NE2", "OE1"],
    "R": ["CA", "N", "C", "CB", "O", "CG", "CD", "CZ", "NE", "NH1", "NH2"],
    "S": ["CA", "N", "C", "CB", "O", "OG"],
    "T": ["CA", "N", "C", "CB", "O", "CG2", "OG1"],
    "V": ["CA", "N", "C", "CB", "O", "CG1", "CG2"],
    "W": ["CA", "N", "C", "CB", "O", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"],
    "Y": ["CA", "N", "C", "CB", "O", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "-": [],
}

# Mask atoms if they are not in the given residue.
residue_atoms_mask: Dict[str, List[bool]] = {
    res: len(residue_atoms[res]) * [True] + (14 - len(residue_atoms[res])) * [False]
    for res in residue_atoms
}


# List of atom types.
atom_types: List[str] = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]


# 2. Van der Waals radii constants.

# Van der Waals radius of each atomic element.
van_der_waals_radius: Dict[str, float] = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}

# List of Van der Waals radii of each residue's atoms.
# If atom not present for a residue, set to zero.
residue_van_der_waals_radius = {
    x: [van_der_waals_radius[atom[0]] for atom in residue_atoms[x]]
    + [0] * (14 - len(residue_atoms[x]))
    for x in residue_atoms
}

# Between-residue bond lengths for general bonds (first element) and for Proline
# (second element).
between_res_bond_length_c_n = [1.329, 1.341]
between_res_bond_length_stddev_c_n = [0.014, 0.016]

# 3. Frame and torsion angles constants.

# Position of atoms in each ref frame.
rigid_group_atom_positions2: Dict[str, Dict[str, List[Union[int, Tuple[float, float, float]]]]] = {
    "A": {
        "C": [0, (1.526, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.529, -0.774, -1.205)],
        "N": [0, (-0.525, 1.363, 0.0)],
        "O": [3, (-0.627, 1.062, 0.0)],
    },
    "C": {
        "C": [0, (1.524, 0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.519, -0.773, -1.212)],
        "N": [0, (-0.522, 1.362, -0.0)],
        "O": [3, (-0.625, 1.062, -0.0)],
        "SG": [4, (-0.728, 1.653, 0.0)],
    },
    "D": {
        "C": [0, (1.527, 0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.526, -0.778, -1.208)],
        "CG": [4, (-0.593, 1.398, -0.0)],
        "N": [0, (-0.525, 1.362, -0.0)],
        "O": [3, (-0.626, 1.062, -0.0)],
        "OD1": [5, (-0.61, 1.091, 0.0)],
        "OD2": [5, (-0.592, -1.101, 0.003)],
    },
    "E": {
        "C": [0, (1.526, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.526, -0.781, -1.207)],
        "CD": [5, (-0.6, 1.397, 0.0)],
        "CG": [4, (-0.615, 1.392, 0.0)],
        "N": [0, (-0.528, 1.361, 0.0)],
        "O": [3, (-0.626, 1.062, 0.0)],
        "OE1": [6, (-0.607, 1.095, -0.0)],
        "OE2": [6, (-0.589, -1.104, 0.001)],
    },
    "F": {
        "C": [0, (1.524, 0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.525, -0.776, -1.212)],
        "CD1": [5, (-0.709, 1.195, -0.0)],
        "CD2": [5, (-0.706, -1.196, 0.0)],
        "CE1": [5, (-2.102, 1.198, -0.0)],
        "CE2": [5, (-2.098, -1.201, -0.0)],
        "CG": [4, (-0.607, 1.377, 0.0)],
        "CZ": [5, (-2.794, -0.003, 0.001)],
        "N": [0, (-0.518, 1.363, 0.0)],
        "O": [3, (-0.626, 1.062, -0.0)],
    },
    "G": {
        "C": [0, (1.517, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "N": [0, (-0.572, 1.337, 0.0)],
        "O": [3, (-0.626, 1.062, -0.0)],
    },
    "H": {
        "C": [0, (1.525, 0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.525, -0.778, -1.208)],
        "CD2": [5, (-0.889, -1.021, -0.003)],
        "CE1": [5, (-2.03, 0.851, -0.002)],
        "CG": [4, (-0.6, 1.37, -0.0)],
        "N": [0, (-0.527, 1.36, 0.0)],
        "ND1": [5, (-0.744, 1.16, -0.0)],
        "NE2": [5, (-2.145, -0.466, -0.004)],
        "O": [3, (-0.625, 1.063, 0.0)],
    },
    "I": {
        "C": [0, (1.527, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.536, -0.793, -1.213)],
        "CD1": [5, (-0.619, 1.391, 0.0)],
        "CG1": [4, (-0.534, 1.437, -0.0)],
        "CG2": [4, (-0.54, -0.785, 1.199)],
        "N": [0, (-0.493, 1.373, -0.0)],
        "O": [3, (-0.627, 1.062, -0.0)],
    },
    "K": {
        "C": [0, (1.526, 0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.524, -0.778, -1.208)],
        "CD": [5, (-0.559, 1.417, 0.0)],
        "CE": [6, (-0.56, 1.416, 0.0)],
        "CG": [4, (-0.619, 1.39, 0.0)],
        "N": [0, (-0.526, 1.362, -0.0)],
        "NZ": [7, (-0.554, 1.387, 0.0)],
        "O": [3, (-0.626, 1.062, -0.0)],
    },
    "L": {
        "C": [0, (1.525, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.522, -0.773, -1.214)],
        "CD1": [5, (-0.53, 1.43, -0.0)],
        "CD2": [5, (-0.535, -0.774, -1.2)],
        "CG": [4, (-0.678, 1.371, 0.0)],
        "N": [0, (-0.52, 1.363, 0.0)],
        "O": [3, (-0.625, 1.063, -0.0)],
    },
    "M": {
        "C": [0, (1.525, 0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.523, -0.776, -1.21)],
        "CE": [6, (-0.32, 1.786, -0.0)],
        "CG": [4, (-0.613, 1.391, -0.0)],
        "N": [0, (-0.521, 1.364, -0.0)],
        "O": [3, (-0.625, 1.062, -0.0)],
        "SD": [5, (-0.703, 1.695, 0.0)],
    },
    "N": {
        "C": [0, (1.526, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.531, -0.787, -1.2)],
        "CG": [4, (-0.584, 1.399, 0.0)],
        "N": [0, (-0.536, 1.357, 0.0)],
        "ND2": [5, (-0.593, -1.188, -0.001)],
        "O": [3, (-0.625, 1.062, 0.0)],
        "OD1": [5, (-0.633, 1.059, 0.0)],
    },
    "P": {
        "C": [0, (1.527, -0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.546, -0.611, -1.293)],
        "CD": [5, (-0.477, 1.424, 0.0)],
        "CG": [4, (-0.382, 1.445, 0.0)],
        "N": [0, (-0.566, 1.351, -0.0)],
        "O": [3, (-0.621, 1.066, 0.0)],
    },
    "Q": {
        "C": [0, (1.526, 0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.525, -0.779, -1.207)],
        "CD": [5, (-0.587, 1.399, -0.0)],
        "CG": [4, (-0.615, 1.393, 0.0)],
        "N": [0, (-0.526, 1.361, -0.0)],
        "NE2": [6, (-0.593, -1.189, 0.001)],
        "O": [3, (-0.626, 1.062, -0.0)],
        "OE1": [6, (-0.634, 1.06, 0.0)],
    },
    "R": {
        "C": [0, (1.525, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.524, -0.778, -1.209)],
        "CD": [5, (-0.564, 1.414, 0.0)],
        "CG": [4, (-0.616, 1.39, -0.0)],
        "CZ": [7, (-0.758, 1.093, -0.0)],
        "N": [0, (-0.524, 1.362, -0.0)],
        "NE": [6, (-0.539, 1.357, -0.0)],
        "NH1": [7, (-0.206, 2.301, 0.0)],
        "NH2": [7, (-2.078, 0.978, -0.0)],
        "O": [3, (-0.626, 1.062, 0.0)],
    },
    "S": {
        "C": [0, (1.525, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.518, -0.777, -1.211)],
        "N": [0, (-0.529, 1.36, -0.0)],
        "O": [3, (-0.626, 1.062, -0.0)],
        "OG": [4, (-0.503, 1.325, 0.0)],
    },
    "T": {
        "C": [0, (1.526, 0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.516, -0.793, -1.215)],
        "CG2": [4, (-0.55, -0.718, 1.228)],
        "N": [0, (-0.517, 1.364, 0.0)],
        "O": [3, (-0.626, 1.062, 0.0)],
        "OG1": [4, (-0.472, 1.353, 0.0)],
    },
    "V": {
        "C": [0, (1.527, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.533, -0.795, -1.213)],
        "CG1": [4, (-0.54, 1.429, -0.0)],
        "CG2": [4, (-0.533, -0.776, -1.203)],
        "N": [0, (-0.494, 1.373, -0.0)],
        "O": [3, (-0.627, 1.062, -0.0)],
    },
    "W": {
        "C": [0, (1.525, -0.0, 0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.523, -0.776, -1.212)],
        "CD1": [5, (-0.824, 1.091, 0.0)],
        "CD2": [5, (-0.854, -1.148, -0.005)],
        "CE2": [5, (-2.186, -0.678, -0.007)],
        "CE3": [5, (-0.622, -2.53, -0.007)],
        "CG": [4, (-0.609, 1.37, -0.0)],
        "CH2": [5, (-3.028, -2.89, -0.013)],
        "CZ2": [5, (-3.283, -1.543, -0.011)],
        "CZ3": [5, (-1.715, -3.389, -0.011)],
        "N": [0, (-0.521, 1.363, 0.0)],
        "NE1": [5, (-2.14, 0.69, -0.004)],
        "O": [3, (-0.627, 1.062, 0.0)],
    },
    "Y": {
        "C": [0, (1.524, -0.0, -0.0)],
        "CA": [0, (0.0, 0.0, 0.0)],
        "CB": [0, (-0.522, -0.776, -1.213)],
        "CD1": [5, (-0.716, 1.195, -0.0)],
        "CD2": [5, (-0.713, -1.194, -0.001)],
        "CE1": [5, (-2.107, 1.2, -0.002)],
        "CE2": [5, (-2.104, -1.201, -0.003)],
        "CG": [4, (-0.607, 1.382, -0.0)],
        "CZ": [5, (-2.791, -0.001, -0.003)],
        "N": [0, (-0.522, 1.362, 0.0)],
        "O": [3, (-0.627, 1.062, -0.0)],
        "OH": [5, (-4.168, -0.002, -0.005)],
    },
    "-": {
        "CA": [0, (0.0, 0.0, 0.0)],
    },
}

# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
chi_angles_atoms: Dict[str, List[Any]] = {
    "A": [],
    "C": [["N", "CA", "CB", "SG"]],
    "D": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "E": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "F": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "G": [],
    "H": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "I": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "K": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "L": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "M": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"]],
    "N": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "P": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "Q": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "R": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "S": [["N", "CA", "CB", "OG"]],
    "T": [["N", "CA", "CB", "OG1"]],
    "V": [["N", "CA", "CB", "CG1"]],
    "W": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "Y": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "-": [],
}

chi_angles_atoms_frames = {
    res: [[residue_atoms[res].index(atom) for atom in angle[1:]] for angle in chi_angles_atoms[res]]
    for res in chi_angles_atoms
}
for res in chi_angles_atoms_frames:
    chi_angles_atoms_frames[res] = chi_angles_atoms_frames[res] + (
        4 - len(chi_angles_atoms_frames[res])
    ) * [[13, 13, 13]]


all_frames = {
    res: [True] + len(chi_angles_atoms[res]) * [True] + (4 - len(chi_angles_atoms[res])) * [False]
    for res in chi_angles_atoms
}
all_frames_atoms = {
    res: [[1, 0, 2]] + chi_angles_atoms_frames[res] for res in chi_angles_atoms_frames
}

all_frames["-"] = [False] * 5

# Number of rigid frames for each residue.
valid_rigids = {x: len(chi_angles_atoms[x]) + 2 for x in chi_angles_atoms}

# Atoms indexes for each residue chi torsion angles.
chi_angles_positions: Dict[str, List[Any]] = {}
for r in residue_atoms:
    chi_angles_positions[r] = []
    for angs in chi_angles_atoms[r]:
        chi_angles_positions[r].append([residue_atoms[r].index(atom) for atom in angs])
    chi_angles_positions[r] += [[13, 13, 13, 13]] * (4 - len(chi_angles_positions[r]))

# Chi-2 torsion angle center for each residue.
chi2_centers: Dict[str, str] = {
    x: chi_angles_atoms[x][1][-2] if len(chi_angles_atoms[x]) > 1 else "CA"
    for x in chi_angles_atoms
}

# Chi-3 torsion angle center for each residue.
chi3_centers: Dict[str, str] = {
    x: chi_angles_atoms[x][2][-2] if len(chi_angles_atoms[x]) > 2 else "CA"
    for x in chi_angles_atoms
}

# Chi-4 torsion angle center for each residue.
chi4_centers: Dict[str, str] = {
    x: chi_angles_atoms[x][3][-2] if len(chi_angles_atoms[x]) > 3 else "CA"
    for x in chi_angles_atoms
}

# Local positions of each atom involved in a residue's torsion angles.
rel_pos = {
    x: [
        rigid_group_atom_positions2[x][residue_atoms[x][atom_id]]
        if len(residue_atoms[x]) > atom_id
        else [0, (0, 0, 0)]
        for atom_id in range(14)
    ]
    for x in rigid_group_atom_positions2
}
