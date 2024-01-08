"""Classes and methods for rigid body operations.

Code adapted and improved from https://github.com/oxpig/ImmuneBuilder/blob/main/ImmuneBuilder/rigids.py

Handles batches, vectorization for fast compute, improved documentation and enhanced features.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from lightmhc.util.constants import (
    chi2_centers,
    chi3_centers,
    chi4_centers,
    rel_pos,
    residue_atoms_mask,
    rigid_group_atom_positions2,
)


class Vector:
    """Class that reimplements atomic coordinates operations for rigid-body updates."""

    def __init__(self, tensor: torch.Tensor) -> None:
        """Initialize a vector of atomic coordinates. Takes as input coordinates for each axis.

        Args:
            tensor: Tensor containing the vector. Shape = (..., 3).
        """
        self.tensor = tensor
        self.shape = tensor.shape
        self.requires_grad = tensor.requires_grad

    def __str__(self) -> str:
        """Return a representation of a Vector instance."""
        return (
            f"Vector(x={self.tensor[..., 0]},\ny={self.tensor[..., 1]},\nz={self.tensor[..., 2]})\n"
        )

    def __repr__(self) -> str:
        """Return a representation of a Vector instance."""
        return str(self)

    def __eq__(self, other: object) -> bool:
        """Returns whether two vectors are equal."""
        if not isinstance(other, Vector):
            return NotImplemented
        return torch.equal(self.tensor, other.tensor)

    def __add__(self, vec: Vector) -> Vector:
        """Compute sum of two vectors.

        Args:
            vec: Vector added to self.

        Returns:
            Vector with summed coordinates.
        """
        return Vector(self.tensor + vec.tensor)

    def __sub__(self, vec: Vector) -> Vector:
        """Compute substraction of two vectors.

        Args:
            vec: Vector substracted to self.

        Returns:
            Vector with substracted coordinates.
        """
        return Vector(self.tensor - vec.tensor)

    def __mul__(self, param: float | torch.Tensor | Vector) -> Vector:
        """Compute multiplication of a vector with a scalar.

        Args:
            param: Scalar multiplied with self.

        Returns:
            Vector with coordinates multiplied by a scalar.
        """
        if isinstance(param, (float, torch.Tensor)):
            return Vector(param * self.tensor)
        if isinstance(param, Vector):
            return Vector(param.tensor * self.tensor)

        raise TypeError("Param must be a scalar.")

    def __matmul__(self, vec: Vector) -> torch.Tensor:
        """Compute dot product between self and vec.

        Args:
            vec: Vector multiplied with self.

        Returns:
            Dot product of self and vec.
        """
        return torch.mul(self.tensor, vec.tensor).sum(-1)

    def __setitem__(self, key: int | tuple[int | slice | list[int], ...], value: Vector) -> None:
        """Replace vector coordinates of a given index by the coordinates of another vector.

        Args:
            key: Index where to modify the values.
            value: Vector containing new coordinates.
        """
        self.tensor[key] = value.tensor

    def __getitem__(self, key: int | tuple[int | slice | list[int], ...]) -> Vector:
        """Return vector coordinates of a given index.

        Args:
            key: Index.

        Returns:
            Vector coordinates of a given index.
        """
        return Vector(self.tensor[key])

    def norm(self) -> torch.Tensor:
        """Computes Euclidean norm of a vector.

        Returns:
            Euclidean norm of self.
        """
        return torch.norm(self.tensor, dim=-1)

    def cross(self, other: Vector) -> Vector:
        """Compute cross product with another vector.

        Args:
            other: Vector to which self is multiplied.

        Returns:
            Cross-product of self and other.
        """
        return Vector(torch.cross(self.tensor, other.tensor, dim=-1))

    def dist(self, other: Vector) -> torch.Tensor:
        """Compute distance between two vectors.

        Args:
            other: Vector to which distance is computed.

        Returns:
            Distance between self and other.
        """
        return torch.cdist(self.tensor, other.tensor)

    def expand(self, shape: list[int] | tuple[int, ...]) -> Vector:
        """Expand along given dimensions.

        Args:
            shape: Expanded dimensions.

        Returns:
            Vector instance with expanded tensors.
        """
        shape = list(shape) + [-1]
        return Vector(self.tensor.expand(shape))

    def unsqueeze(self, dim: int) -> Vector:
        """Returns a new Vector with a dimension of size one inserted at the specified position.

        Args:
            dim: Position where new dimension is inserted.

        Returns:
            Unsqueezed vector.
        """
        if dim < 0:
            dim -= 1
        return Vector(self.tensor.unsqueeze(dim))

    def squeeze(self, dim: int) -> Vector:
        """Returns a new Vector instance with a dimension of size one removed at the specified position.

        Args:
            dim: Position where new dimension is removed.

        Returns:
            Squeezed Vector instance.
        """
        if dim < 0:
            dim -= 1
        return Vector(self.tensor.squeeze(dim))

    def map_func(self, func: Callable) -> Vector:
        """Operate a function coordinate-wise on a vector.

        Args:
            func: Function called on each coordinate.

        Returns:
            Mapped vector.
        """
        tensor = torch.clone(self.tensor)
        tensor[..., 0] = func(self.tensor[..., 0])
        tensor[..., 1] = func(self.tensor[..., 1])
        tensor[..., 2] = func(self.tensor[..., 2])

        return Vector(tensor)

    def permute(self, shape: list[int] | tuple[int, ...]) -> Vector:
        """Permute according to given dimensions.

        Args:
            shape: Permutation dimensions.

        Returns:
            Permuted vector.
        """
        shape = list(shape) + [-1]
        return Vector(self.tensor.permute(shape))

    def to(self, device: torch.device) -> Vector:
        """Performs device conversion.

        Args:
            device: Device where Vector instance is stored.

        Returns:
            Same instance stored on designated device.
        """
        return Vector(self.tensor.to(device))

    def view(self, shape: tuple[int, ...]) -> Vector:
        """Returns a different view of the Vector.

        Args:
            shape: New vector shape.

        Returns:
            Vector with new shape.
        """
        return Vector(self.tensor.view(list(shape) + [3]))


class Rot:
    """Class that implements 3D rotation matrix operations for rigid-body updates."""

    def __init__(self, tensor: torch.Tensor) -> None:
        """Initialize a 3D-rotation matrix. Needs as input each coefficient of the matrix.

        Args:
            tensor: Tensor containing the rotation matrix. Shape = (..., 3, 3).
        """
        self.tensor = tensor
        self.shape = tensor.shape
        self.requires_grad = tensor.requires_grad

    def __str__(self) -> str:
        """Return a representation of a Rotation instance."""
        return (
            "Rot(xx={},\nxy={},\nxz={},\nyx={},\nyy={},\nyz={},\nzx={},\nzy={},\nzz={})\n".format(
                self.tensor[..., 0, 0],
                self.tensor[..., 0, 1],
                self.tensor[..., 0, 2],
                self.tensor[..., 1, 0],
                self.tensor[..., 1, 1],
                self.tensor[..., 1, 2],
                self.tensor[..., 2, 0],
                self.tensor[..., 2, 1],
                self.tensor[..., 2, 2],
            )
        )

    def __repr__(self) -> str:
        """Return a representation of a Rotation instance."""
        return str(self)

    def __eq__(self, other: object) -> bool:
        """Returns whether two rotations are equal."""
        if not isinstance(other, Rot):
            return NotImplemented
        return torch.equal(self.tensor, other.tensor)

    def __matmul__(self, other: Any) -> Vector | Rot:
        """Rotation matrix multiplication with a vector or another rotation matrix.

        Args:
            other: Vector or Rotation matrix that is multiplied to self.

        Returns:
            Product of the matrix multiplication.
        """
        if isinstance(other, Vector):
            return Vector(torch.matmul(self.tensor, other.tensor.unsqueeze(-1)).squeeze(-1))

        if isinstance(other, Rot):
            return Rot(torch.matmul(self.tensor, other.tensor))

        raise ValueError(f"Matmul against {type(other)}")

    def __setitem__(self, key: int | tuple[int | slice | list[int], ...], value: Rot) -> None:
        """Replace rotation matrix coefficients of a given index by the coefficients of another matrix.

        Args:
            key: Index where to modify the values.
            value: Rotation matrix containing new coefficients.
        """
        self.tensor[key] = value.tensor

    def __getitem__(self, key: int | tuple[int | slice | list[int], ...]) -> Rot:
        """Return rotation matrix of a given index.

        Args:
            key: Index.

        Returns:
            Rotation matrix of a given index.
        """
        return Rot(self.tensor[key])

    def inv(self) -> Rot:
        """Transpose / inverse the rotation matrix.

        Returns:
            Transposed / inversed rotation matrix.
        """
        return Rot(torch.transpose(self.tensor, -2, -1))

    def expand(self, shape: list[int] | tuple[int, ...]) -> Rot:
        """Expand along given dimensions.

        Args:
            shape: Expanded dimensions.

        Returns:
            Rotation instance with expanded tensors.
        """
        shape = list(shape) + [-1, -1]
        return Rot(self.tensor.expand(shape))

    def map_func(self, func: Callable) -> Rot:
        """Operate a function coordinate-wise on a rotation matrix.

        Args:
            func: Function called on each coordinate.

        Returns:
            Mapped rotation matrix.
        """
        return Rot(func(self.tensor))

    def unsqueeze(self, dim: int) -> Rot:
        """Returns a new Rotation with a dimension of size one inserted at the specified position.

        Operates on each coordinate of the rotation matrix.

        Args:
            dim: Position where new dimension is inserted.

        Returns:
            Unsqueezed rotation matrix.
        """
        if dim < 0:
            dim -= 2
        return Rot(self.tensor.unsqueeze(dim=dim))

    def squeeze(self, dim: int) -> Rot:
        """Returns a new Rotation instance with a dimension of size one removed at the specified position.

        Operates on each coordinate of the rotation matrix.

        Args:
            dim: Position where new dimension is removed.

        Returns:
            Squeezed Rotation instance.
        """
        if dim < 0:
            dim -= 2
        return Rot(self.tensor.squeeze(dim=dim))

    def detach(self) -> Rot:
        """Detach from current graph.

        Returns:
            New rotation object detached from current graph.
        """
        return Rot(self.tensor.detach())

    def to(self, device: torch.device) -> Rot:
        """Performs device conversion.

        Args:
            device: Device where Rotation instance is stored.

        Returns:
            Same instance stored on designated device.
        """
        return Rot(self.tensor.to(device))

    def view(self, shape: tuple[int, ...]) -> Rot:
        """Returns a different view of the rotation matrix.

        Args:
            shape: New rotation matrix shape.

        Returns:
            Rotation matrix with new shape.
        """
        return Rot(self.tensor.view(list(shape) + [3, 3]))


class Rigid:
    """Rigid frame object."""

    def __init__(self, origin: Vector, rot: Rot):
        """Initialize rigid frame with origin and rotation matrix.

        Args:
            origin: Origin of the frame.
            rot: Rotation matrix of the frame.
        """
        self.origin = origin
        self.rot = rot
        self.shape = self.origin.shape

    def __matmul__(self, other: object) -> Vector | Rigid:
        """Rigid body matrix multiplication with a vector or another Rigid instance.

        If multiplication with a vector, the vector is rotated by rigid instance rotation matrix
        and translated by the rigid instance origin vector.

        If multiplication with a Rigid instance, the 2nd origin vector is rotated by
        1st rigid instance rotation matrix and translated by the 1st origin vector.
        The two rotation matrices are multiplied.

        Args:
            other: Vector or Rotation matrix that is multiplied to self.

        Returns:
            Product of the Rigid instances multiplication.
        """
        if isinstance(other, Vector):
            mul = self.rot @ other
            if isinstance(mul, Vector):
                return mul + self.origin
        elif isinstance(other, Rigid):
            mul1 = self.rot @ other.origin
            mul2 = self.rot @ other.rot
            if isinstance(mul1, Vector) and isinstance(mul2, Rot):
                return Rigid(mul1 + self.origin, mul2)

        raise TypeError(f"can't multiply rigid by object of type {type(other)}")

    def __eq__(self, other: object) -> bool:
        """Returns whether two rigids are equal."""
        if not isinstance(other, Rigid):
            return NotImplemented
        return self.origin == other.origin and self.rot == other.rot

    def __getitem__(self, key: int | tuple[int | slice | list[int], ...]) -> Rigid:
        """Return Rigid instance of a given index.

        Args:
            key: Index.

        Returns:
            Rigid instance of a given index.
        """
        return Rigid(self.origin[key], self.rot[key])

    def __setitem__(self, key: int | tuple[int | slice | list[int], ...], value: Rigid) -> None:
        """Replace Rigid coefficients of a given index by the coefficients of another rigid frame.

        Args:
            key: Index where to modify the values.
            value: Rigid frame containing new coefficients.
        """
        self.origin[key] = value.origin
        self.rot[key] = value.rot

    def inv(self) -> Rigid:
        """Invert Rigid instance.

        Invert rotation matrix is calculated and applied to the O-symmetric origin vector.

        Returns:
            Inverted Rigid instance.
        """
        inv_rot = self.rot.inv()
        t = inv_rot @ self.origin
        if isinstance(t, Vector):
            return Rigid(Vector(-t.tensor), inv_rot)
        raise TypeError("t should be a vector.")

    def expand(self, shape: list[int] | tuple[int, ...]) -> Rigid:
        """Expand along given dimensions.

        Args:
            shape: Expanded dimensions.

        Returns:
            Rigid instance with expanded tensors.
        """
        return Rigid(self.origin.expand(shape), self.rot.expand(shape))

    def map_func(self, func: Callable) -> Rigid:
        """Operate a function coordinate-wise on a rotation matrix and origin vector.

        Args:
            func: Function called on each coordinate.

        Returns:
            Mapped Rigid.
        """
        return Rigid(self.origin.map_func(func), self.rot.map_func(func))

    def unsqueeze(self, dim: int) -> Rigid:
        """Returns a new Rigid instance with a dimension of size one inserted at the specified position.

        Operates on each coordinate of the rotation matrix and on the origin vector.

        Args:
            dim: Position where new dimension is inserted.

        Returns:
            Unsqueezed Rigid instance.
        """
        return Rigid(self.origin.unsqueeze(dim=dim), self.rot.unsqueeze(dim=dim))

    def squeeze(self, dim: int) -> Rigid:
        """Returns a new Rigid instance with a dimension of size one removed at the specified position.

        Operates on each coordinate of the rotation matrix and on the origin vector.

        Args:
            dim: Position where new dimension is removed.

        Returns:
            Squeezed Rigid instance.
        """
        return Rigid(self.origin.squeeze(dim=dim), self.rot.squeeze(dim=dim))

    def to(self, device: torch.device) -> Rigid:
        """Performs device conversion.

        Args:
            device: Device where Rigid instance is stored.

        Returns:
            Same instance stored on designated device.
        """
        return Rigid(self.origin.to(device), self.rot.to(device))

    def view(self, shape: tuple[int, ...]) -> Rigid:
        """Returns a different view of the rigid frame.

        Args:
            shape: New rigid frame shape.

        Returns:
            Rigid frame with new shape.
        """
        return Rigid(self.origin.view(shape), self.rot.view(shape))


def rigid_body_identity(shape: tuple[int | slice, ...]) -> Rigid:
    """Create an identity element (stable under Rigid multiplication).

    Origin is set to zero and rotation matrix to I3.

    Args:
        shape: Shape of the origin vector.

    Returns:
        Identity Rigid element.
    """
    origin = torch.zeros(3).repeat(list(shape) + [1])
    origin.requires_grad = True
    rot = torch.eye(3).repeat(list(shape) + [1, 1])
    rot.requires_grad = True
    return Rigid(
        Vector(origin),
        Rot(rot),
    )


def rigid_from_three_points(x1: Vector, x2: Vector, x3: Vector) -> Rigid:
    """Create Rigid instance based on 3 points.

    Args:
        x1: Coordinates of the 2nd point - (xy) plane.
        x2: Origin coordinates.
        x3: Coordinates of the 3rd point - (xz) plane.

    Returns:
        Rigid object based on these 3 points.
    """
    v1 = x3 - x2
    v2 = x1 - x2
    v1 = v1 * (1 / v1.norm().unsqueeze(-1))
    v2 = v2 - v1 * (v1 @ v2).unsqueeze(-1)
    v2 *= 1 / v2.norm().unsqueeze(-1)
    v3 = v1.cross(v2)
    rot = Rot(torch.stack([v1.tensor, v2.tensor, v3.tensor], dim=-1))
    return Rigid(x2, rot)


def rigid_from_tensor(tens: torch.Tensor) -> Rigid:
    """Create Rigid instance based on a tensor.

    Args:
        tens: PyTorch tensor containing coordinates of the 3 points.

    Returns:
        Rigid object based on the tensor of coordinates.
    """
    assert tens.shape[-1] == 3, "I want 3D points"
    return rigid_from_three_points(
        Vector(tens[..., 0, :]),
        Vector(tens[..., 1, :]),
        Vector(tens[..., 2, :]),
    )


def stack_rigids(rigids: list[Rigid], dim_vec: int, dim_rot: int) -> Rigid:
    """Stack Rigid objects by stacking each coordinates of origin vector and rotation matrix.

    Args:
        rigids: List of Rigid instances.
        dim_vec: Stack dimension of vectors.
        dim_rot: Stack dimension of rotations.

    Returns:
        Stacked Rigid instance.
    """
    # Probably best to avoid using very much
    stacked_origin = Vector(
        torch.stack([rig.origin.tensor for rig in rigids], dim=dim_vec),
    )
    stacked_rot = Rot(
        torch.stack([rig.rot.tensor for rig in rigids], dim=dim_rot),
    )
    return Rigid(stacked_origin, stacked_rot)


def rotate_x_axis_to_new_vector(new_vector: torch.Tensor) -> Rigid:
    """Rotate a frame such that the new frame x-axis is given by a chosen vector.

    Args:
        new_vector: Vector to be chosen as x-axis.

    Returns:
        Rigid object to change x-axis to chosen vector.
    """
    # Extract coordinates
    c, b, a = new_vector[..., 0], new_vector[..., 1], new_vector[..., 2]

    # Normalize
    n = (c**2 + a**2 + b**2 + 1e-16) ** (1 / 2)
    a, b, c = a / n, b / n, -c / n

    # Set new origin
    new_origin = Vector(torch.zeros_like(new_vector))

    # Rotate x-axis to point old origin to new one
    k = (1 - c) / (a**2 + b**2 + 1e-8)
    shape = list(c.shape) + [3, 3]
    rot = torch.stack(
        [-c, b, -a, b, 1 - k * b**2, a * b * k, a, -a * b * k, k * a**2 - 1], dim=-1
    ).view(shape)
    new_rot = Rot(rot)
    return Rigid(new_origin, new_rot)


def rigid_transformation_from_torsion_angles(
    torsion_angles: torch.Tensor, distance_to_new_origin: torch.Tensor
) -> Rigid:
    """Convert torsion angles to a Rigid instance by rotating around the x-axis.

    Algorithm 25 in AF2 supplementary.
    Args:
        torsion_angles: Tensor containing torsion angle values.
        distance_to_new_origin: Translation to new origin.

    Returns:
        Rigid instance with rotation around the x-axis by torsion angle and translation along x.
    """
    dev = torsion_angles.device

    rot = torch.zeros(list(torsion_angles.shape[:-1]) + [3, 3]).to(dev)
    rot[..., 0, 0] = -torch.ones(torsion_angles.shape[:-1]).to(dev)
    rot[..., 0, 1] = torch.zeros(torsion_angles.shape[:-1]).to(dev)
    rot[..., 0, 2] = torch.zeros(torsion_angles.shape[:-1]).to(dev)
    rot[..., 1, 0] = torch.zeros(torsion_angles.shape[:-1]).to(dev)

    rot[..., 1, 1] = torsion_angles[..., 0]
    rot[..., 1, 2] = torsion_angles[..., 1]
    rot[..., 2, 0] = torch.zeros(torsion_angles.shape[:-1]).to(dev)
    rot[..., 2, 1] = torsion_angles[..., 1]
    rot[..., 2, 2] = -torsion_angles[..., 0]

    new_rot = Rot(rot)
    new_origin = Vector(
        torch.stack(
            [
                distance_to_new_origin,
                torch.zeros(torsion_angles.shape[:-1]).to(dev),
                torch.zeros(torsion_angles.shape[:-1]).to(dev),
            ],
            dim=-1,
        )
    )

    return Rigid(new_origin, new_rot)


def global_frames_from_bb_frame_and_torsion_angles(
    bb_frame: Rigid, torsion_angles: torch.Tensor, sequences: list[str]
) -> Rigid:
    """Compose predicted backbone frame with local side chain reference frames and predicted torsion angles.

    Recovers global all side chains global frames.
    Follows Algorithm 24 in AF2 supplementary.

    Args:
        bb_frame: Predicted backbone frames. Shape = (batch, n_aa).
        torsion_angles: Predicted torsion angles. Shape = (batch, n_aa, 6, 2).
        sequences: List of sequences in the batch. Shape = (batch, n_aa).

    Returns:
        Rigid instance with backbone and side chains predicted global frames.
    """
    dev = bb_frame.origin.tensor.device

    # We start with psi
    psi_local_frame_origin = (
        torch.tensor([[rel_pos[res][2][1] for res in seq] for seq in sequences])
        .to(dev)
        .square()
        .sum(-1)
        .sqrt()
    )
    psi_local_frame = rigid_transformation_from_torsion_angles(
        torsion_angles[..., 0, :], psi_local_frame_origin
    )
    psi_global_frame = bb_frame @ psi_local_frame

    # Now all the chis
    chi1_local_frame_origin = torch.tensor(
        [[rel_pos[res][3][1] for res in seq] for seq in sequences]
    ).to(dev)
    chi1_local_frame = rotate_x_axis_to_new_vector(
        chi1_local_frame_origin
    ) @ rigid_transformation_from_torsion_angles(
        torsion_angles[..., 1, :], chi1_local_frame_origin.square().sum(-1).sqrt()
    )
    chi1_global_frame = bb_frame @ chi1_local_frame

    chi2_local_frame_origin = torch.tensor(
        [
            [rigid_group_atom_positions2[res][chi2_centers[res]][1] for res in seq]
            for seq in sequences
        ]
    ).to(dev)
    chi2_local_frame = rotate_x_axis_to_new_vector(
        chi2_local_frame_origin
    ) @ rigid_transformation_from_torsion_angles(
        torsion_angles[..., 2, :], chi2_local_frame_origin.square().sum(-1).sqrt()
    )
    if isinstance(chi1_global_frame, Rigid):
        chi2_global_frame = chi1_global_frame @ chi2_local_frame

        chi3_local_frame_origin = torch.tensor(
            [
                [rigid_group_atom_positions2[res][chi3_centers[res]][1] for res in seq]
                for seq in sequences
            ]
        ).to(dev)
        chi3_local_frame = rotate_x_axis_to_new_vector(
            chi3_local_frame_origin
        ) @ rigid_transformation_from_torsion_angles(
            torsion_angles[..., 3, :], chi3_local_frame_origin.square().sum(-1).sqrt()
        )
        if isinstance(chi2_global_frame, Rigid):

            chi3_global_frame = chi2_global_frame @ chi3_local_frame

            chi4_local_frame_origin = torch.tensor(
                [
                    [rigid_group_atom_positions2[res][chi4_centers[res]][1] for res in seq]
                    for seq in sequences
                ]
            ).to(dev)
            chi4_local_frame = rotate_x_axis_to_new_vector(
                chi4_local_frame_origin
            ) @ rigid_transformation_from_torsion_angles(
                torsion_angles[..., 4, :], chi4_local_frame_origin.square().sum(-1).sqrt()
            )
            if isinstance(chi3_global_frame, Rigid):
                chi4_global_frame = chi3_global_frame @ chi4_local_frame
                if (
                    isinstance(bb_frame, Rigid)
                    and isinstance(psi_global_frame, Rigid)
                    and isinstance(chi1_global_frame, Rigid)
                    and isinstance(chi2_global_frame, Rigid)
                    and isinstance(chi3_global_frame, Rigid)
                    and isinstance(chi4_global_frame, Rigid)
                ):
                    return stack_rigids(
                        [
                            bb_frame,
                            psi_global_frame,
                            chi1_global_frame,
                            chi2_global_frame,
                            chi3_global_frame,
                            chi4_global_frame,
                        ],
                        dim_vec=-2,
                        dim_rot=-3,
                    )
    raise TypeError("All frames must be Rigid.")


def all_atoms_from_global_reference_frames(
    global_reference_frames: Rigid, sequences: list[str]
) -> torch.Tensor:
    """Convert local initial atom positions into global coordinates using global reference frames.

    Args:
        global_reference_frames: Predicted global frames (trans, rot) for each atom.
                                 Shape = (batch, n_aa, 6).
        sequences: List of sequences in the batch. Shape = (batch, n_aa).

    Returns:
        Coordinates of all atoms. Shape = (batch, n_aa, 14, 3).
    """
    dev = global_reference_frames.origin.tensor.device

    relative_positions = [
        [[rel_pos[res][atom_pos][1] for atom_pos in range(14)] for res in seq] for seq in sequences
    ]
    local_reference_frame = torch.tensor([[[max(rel_pos[res][atom_pos][0] - 2, 0) for atom_pos in range(14)] for res in seq] for seq in sequences])  # type: ignore
    local_reference_frame_mask = torch.stack(
        [y == local_reference_frame for y in range(6)], dim=3
    ).to(dev)
    global_reference_frames = global_reference_frames.unsqueeze(-2).expand((-1, -1, 14, -1))[
        local_reference_frame_mask
    ]
    global_reference_frames = global_reference_frames.view((len(sequences), len(sequences[0]), 14))
    global_atom_vector = global_reference_frames @ Vector(
        torch.tensor(relative_positions).to(dev).to(torch.float32)
    )
    if isinstance(global_atom_vector, Vector):
        all_atoms = global_atom_vector.tensor
    else:
        raise TypeError("Global_atom_vector must be a Vector.")

    all_atom_mask = torch.tensor(
        [[residue_atoms_mask[res] for res in seq] for seq in sequences]
    ).to(dev)
    all_atoms[~all_atom_mask] = float("Nan")
    return all_atoms
