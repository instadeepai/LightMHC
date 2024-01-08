"""Test functions of rigids."""
from typing import Callable

import numpy as np
import pytest
import torch

from lightmhc.rigids import (
    Rigid,
    Rot,
    Vector,
    rigid_body_identity,
    rigid_from_tensor,
    rigid_from_three_points,
    stack_rigids,
)

shape = 10


@pytest.fixture(scope="module", name="v1")
def fixture_v1() -> Vector:
    """First vector used for tests.

    Returns:
        Vector(1, 1, 1).
    """
    return Vector(torch.ones(shape, 3))


@pytest.fixture(scope="module", name="v2")
def fixture_v2() -> Vector:
    """Second vector used for tests.

    Returns:
        Vector(3, 3, 3).
    """
    return Vector(3 * torch.ones(shape, 3))


@pytest.fixture(scope="module", name="rot1")
def fixture_rot1() -> Rot:
    """First rotation used for tests.

    Returns:
        Rot(1, 0, 0)
           (0, 1, 0)
           (0, 0, 1).
    """
    return Rot(torch.eye(3).repeat(shape, 1, 1))


@pytest.fixture(scope="module", name="rot2")
def fixture_rot2() -> Rot:
    """Second rotation used for tests.

    Returns:
        Rot(1/2,        sqrt(3)/2, 0)
           (-sqrt(3)/2, 1/2,       0)
           (0,          0,         1).
    """
    tensor = torch.tensor(
        [[0.5, 0.5 * (3**0.5), 0.0], [-0.5 * (3**0.5), 0.5, 0.0], [0.0, 0.0, 1.0]]
    ).repeat(shape, 1, 1)
    return Rot(tensor)


@pytest.fixture(scope="module", name="rigid1")
def fixture_rigid1() -> Rigid:
    """First rigid used for tests.

    Returns:
        Vector(1, 1, 1)
        Rot(1, 0, 0)
           (0, 1, 0)
           (0, 0, 1).
    """
    return Rigid(Vector(torch.ones(shape, 3)), Rot(torch.eye(3).repeat(shape, 1, 1)))


@pytest.fixture(scope="module", name="rigid2")
def fixture_rigid2() -> Rigid:
    """Second rigid used for tests.

    Returns:
        Vector(3, 3, 3)
        Rot(1/2,        sqrt(3)/2, 0)
           (-sqrt(3)/2, 1/2,       0)
           (0,          0,         1).
    """
    return Rigid(Vector(3 * torch.ones(shape, 3)), Rot(torch.eye(3).repeat(shape, 1, 1)))


@pytest.fixture(scope="module", name="func")
def fixture_func() -> Callable:
    """Coordinate-wise function used for tests.

    Returns:
        f: function x: -> x+1.
    """
    f: Callable = lambda x: x + 1
    return f


@pytest.fixture(scope="module", name="gamma")
def fixture_gamma() -> float:
    """Gamma coefficient used for tests.

    Returns:
        gamma: 3.0
    """
    return 3.0


class TestVector:
    """Class to test Vector functions."""

    def test_add(self, v1: Vector, v2: Vector) -> None:
        """Test add function.

        Args:
            v1: First vector.
            v2: Second vector.

        Tests:
            Vector coordinates are summed.
        """
        assert v1 + v2 == Vector(4 * torch.ones(shape, 3))

    def test_sub(self, v1: Vector, v2: Vector) -> None:
        """Test sub function.

        Args:
            v1: First vector.
            v2: Second vector.

        Tests:
            Vector coordinates are substracted.
        """
        assert v1 - v2 == Vector(-2 * torch.ones(shape, 3))

    def test_mul(self, v1: Vector, gamma: float) -> None:
        """Test mul function.

        Args:
            v1: Vector to be multiplied.
            gamma: Scalar multiplied to each coordinate.

        Tests:
            Vector coordinates are multiplied by the gamma scalar.
        """
        assert v1 * gamma == Vector(3 * torch.ones(shape, 3))

    def test_matmul(self, v1: Vector, v2: Vector) -> None:
        """Test matmul function.

        Args:
            v1: First vector.
            v2: Second vector.

        Tests:
            Correct dot product is computed.
        """
        assert torch.equal(v1 @ v2, 9 * torch.ones(shape))

    def test_norm(self, v1: Vector) -> None:
        """Test norm function.

        Args:
            v1: Vector.

        Tests:
            Correct Euclidean norm is computed.

        """
        assert torch.equal(v1.norm(), torch.sqrt(3 * torch.ones(shape)))

    def test_cross(self, v1: Vector, v2: Vector) -> None:
        """Test cross function.

        Args:
            v1: First vector.
            v2: Second vector.

        Tests:
            Correct cross product is computed.
        """
        assert v1.cross(v2) == Vector(torch.zeros(shape, 3))

    def test_dist(self, v1: Vector, v2: Vector) -> None:
        """Test dist function.

        Args:
            v1: First vector.
            v2: Second vector.

        Tests:
            Correct distance is computed.
        """
        assert torch.equal(v1.dist(v2), torch.sqrt(12 * torch.ones(shape, shape)))

    def test_expand(self, v1: Vector) -> None:
        """Test expand function.

        Args:
            v1: Vector to be expanded.

        Tests:
            Expanded vector is correct.
        """
        assert v1.expand((2, -1)) == Vector(torch.ones(2, shape, 3))

    def test_unsqueeze(self, v1: Vector) -> None:
        """Test unsqueeze function.

        Args:
            v1: Vector to be unsqueezed.

        Tests:
            Correct shapes after unsqueeze.
        """
        assert v1.unsqueeze(0) == Vector(torch.ones(1, shape, 3))

    def test_squeeze(self, v1: Vector) -> None:
        """Test squeeze function.

        Args:
            v1: Vector to be squeezed.

        Tests:
            After unsqueeze and squeeze, initial vector is returned.
        """
        assert v1.unsqueeze(0).squeeze(0) == v1

    def test_map(self, v1: Vector, func: Callable) -> None:
        """Test map function.

        Args:
            v1: Vector to be mapped.
            func: Function to be mapped.

        Tests:
            Map function is correctly applied on each coordinate.
        """
        assert v1.map_func(func) == Vector(2 * torch.ones(shape, 3))

    def test_permute(self, v1: Vector) -> None:
        """Test permute function.

        Args:
            v1: Vector to be permuted.

        Tests:
            Correct shapes after permutation.
        """
        v = v1.unsqueeze(0)
        assert v.permute((1, 0)).shape == (shape, 1, 3)


class TestRot:
    """Class to test rotation functions."""

    def test_matmul(self, rot1: Rot, rot2: Rot, v1: Vector) -> None:
        """Test matmul function.

        Args:
            rot1: First rotation.
            rot2: Second rotation.
            v1: Vector.

        Tests:
            Result of multiplication by vector and matrix is correct.
        """
        mul1 = rot1 @ v1
        mul2 = rot2 @ rot2

        tensor_rot2 = (
            torch.tensor(
                [[-0.5, 0.5 * np.sqrt(3), 0.0], [-0.5 * np.sqrt(3), -0.5, 0.0], [0.0, 0.0, 1.0]]
            )
            .to(torch.float32)
            .repeat(shape, 1, 1)
        )

        if isinstance(mul1, Vector) and isinstance(mul2, Rot):
            assert mul1 == v1
            assert mul2 == Rot(tensor_rot2)

        else:
            raise TypeError(
                "First (resp. second) multiplication should return a vector (resp. rotation)."
            )

    def test_inv(self, rot1: Rot) -> None:
        """Test inverse function.

        Args:
            rot1: Rotation matrix.

        Tests:
            Correct inversed (transposed) rotation matrix.
        """
        assert rot1.inv() == rot1

    def test_expand(self, rot1: Rot) -> None:
        """Test expand function.

        Args:
            rot1: Rotation to be expanded.

        Tests:
            Expanded rotation is correct.
        """
        assert rot1.expand((2, -1)) == Rot(torch.eye(3).repeat(2, shape, 1, 1))

    def test_map(self, rot1: Rot, func: Callable) -> None:
        """Test map function.

        Args:
            rot1: Rotation to be mapped.
            func: Function to apply to rotation matrix.

        Tests:
            Mapped rotation is correct.
        """
        tensor_rot = torch.ones(shape, 3, 3)
        tensor_rot[:, 0, 0] = 2.0
        tensor_rot[:, 1, 1] = 2.0
        tensor_rot[:, 2, 2] = 2.0

        assert rot1.map_func(func) == Rot(tensor_rot)

    def test_unsqueeze(self, rot1: Rot) -> None:
        """Test unsqueeze function.

        Args:
            rot1: Rotation to be unsqueezed.

        Tests:
            Correct shapes after unsqueeze.
        """
        assert rot1.unsqueeze(0) == Rot(torch.eye(3).repeat(1, shape, 1, 1))

    def test_squeeze(self, rot1: Rot) -> None:
        """Test squeeze function.

        Args:
            rot1: Rotation to be squeezed.

        Tests:
            After unsqueeze and squeeze, initial rotation is returned.
        """
        rot = rot1.unsqueeze(0).squeeze(0)
        assert rot == rot1


class TestRigid:
    """Class to test rigid functions."""

    def test_inv(self, rigid1: Rigid) -> None:
        """Test inverse function.

        Args:
            rigid1: Rigid body.

        Tests:
            Opposite origin.
            Correct inversed (transposed) rotation matrix.
        """
        assert rigid1.inv() == Rigid(Vector(-torch.ones(shape, 3)), rigid1.rot)

    def test_expand(self, rigid1: Rigid) -> None:
        """Test expand function.

        Args:
            rigid1: Rigid to be expanded.

        Tests:
            Expanded rigid is correct.
        """
        assert rigid1.expand((2, -1)) == Rigid(
            Vector(torch.ones(2, shape, 3)), Rot(torch.eye(3).repeat(2, shape, 1, 1))
        )

    def test_map(self, rigid1: Rigid, func: Callable) -> None:
        """Test map function.

        Args:
            rigid1: Rigid body to be mapped.
            func: Function to apply to rotation matrix.

        Tests:
            Mapped rigid body is correct.
        """
        rot_tensor = torch.ones(shape, 3, 3)
        rot_tensor[:, 0, 0] = 2.0
        rot_tensor[:, 1, 1] = 2.0
        rot_tensor[:, 2, 2] = 2.0

        assert rigid1.map_func(func) == Rigid(Vector(2 * torch.ones(shape, 3)), Rot(rot_tensor))

    def test_unsqueeze(self, rigid1: Rigid) -> None:
        """Test unsqueeze function.

        Args:
            rigid1: Rigid to be unsqueezed.

        Tests:
            Correct shapes after unsqueeze.
        """
        assert rigid1.unsqueeze(0).origin.tensor.shape == (1, shape, 3)
        assert rigid1.unsqueeze(0).rot.tensor.shape == (1, shape, 3, 3)

    def test_squeeze(self, rigid1: Rigid) -> None:
        """Test squeeze function.

        Args:
            rigid1: Rigid to be squeezed.

        Tests:
            After unsqueeze and squeeze, shapes are the same as initial rigid1 object.
        """
        assert rigid1.unsqueeze(0).squeeze(0).origin.tensor.shape == (shape, 3)
        assert rigid1.unsqueeze(0).squeeze(0).rot.tensor.shape == (shape, 3, 3)


def test_rigid_body_identity() -> None:
    """Test rigid_body_identity.

    Tests:
        Origin is set to (0,0,0).
        Rotation is set to identity matrix.
    """
    assert rigid_body_identity((shape,)).origin == Vector(torch.zeros(shape, 3))

    assert rigid_body_identity((shape,)).rot == Rot(torch.eye(3).repeat(shape, 1, 1))


def test_rigid_from_three_points() -> None:
    """Test rigid_from_three_points function.

    Uses a specific set of 3 points: (0, 0, 0), (0, 1, 0), (1, 0, 0).

    Tests:
        Origin is set to (0, 0, 0).
        Rotation is set to identity matrix.
    """
    v1 = Vector(torch.zeros(shape, 3))
    v2 = Vector(torch.zeros(shape, 3))
    v1.tensor[:, 1] = 1.0

    v3 = Vector(torch.zeros(shape, 3))
    v3.tensor[:, 0] = 1.0
    rigid = rigid_from_three_points(v1, v2, v3)

    assert rigid.origin == v2
    assert rigid.rot == Rot(torch.eye(3).repeat(shape, 1, 1))


def test_rigid_from_tensor() -> None:
    """Test rigid_from_tensor function.

    Uses a specific set of 3 points: (0, 0, 0), (0, 1, 0), (1, 0, 0).

    Tests:
        Origin is set to (0, 0, 0).
        Rotation is set to identity matrix.
    """
    tens = torch.zeros(shape, 3, 3)
    tens[:, 0, 1] = 1.0
    tens[:, 2, 0] = 1.0
    rigid = rigid_from_tensor(tens)
    assert rigid.origin == Vector(torch.zeros(shape, 3))
    assert rigid.rot == Rot(torch.eye(3).repeat(shape, 1, 1))


def test_stack_rigids(rigid1: Rigid, rigid2: Rigid) -> None:
    """Test stack_rigids function.

    Args:
        rigid1: First rigid body.
        rigid2: Second rigid body.

    Tests:
        Stacked rigid has correct first shape.
    """
    assert stack_rigids([rigid1, rigid2], 0, 0).shape[0] == 2
