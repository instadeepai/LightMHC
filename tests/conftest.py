"""Fixtures and variable used to test different codes in the project."""
# pylint: disable=redefined-outer-name
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
# pylint: disable=unused-argument
from pathlib import Path

import omegaconf
import pytest

pytest_plugin = "data_gen_test_helpers"


@pytest.fixture(scope="session")
def current_path() -> Path:
    """Get current path of testing directory.

    Returns:
        Current directory path.
    """
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary directory to store files for testing.

    Args:
        tmp_path_factory: fixture used to create temporary directories.

    Returns:
        Temporary directory path.
    """
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Creating a temporary directory where we will put our data."""
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module", name="gnn_structure_fixtures")
def fixture_gnn_structure_assets(current_path: Path) -> Path:
    """Path to the gnn_structure fixture assets.

    Args:
        current_path (fixture): Path to the current test suite.

    Returns:
        Path to the gnn_structure fixture assets.
    """
    return current_path / "fixtures"


@pytest.fixture(scope="session", name="gnn_structure_tempdir")
def fixture_gnn_structure_tempdir(tmp_dir: Path) -> Path:
    """Get path to tempdir for gnn_structure test outputs.

    Args:
        tmp_dir (fixture): Tempdir for test outputs.

    Returns:
        Path to gnn_structure test output temporary directory.
    """
    (gnn_structure_tempdir := tmp_dir / Path(__file__).parent.stem).mkdir()
    return gnn_structure_tempdir


@pytest.fixture(scope="session", name="test_config")
def fixture_gnn_test_config() -> omegaconf.DictConfig:
    """Dummy test configuration used in several tests.

    Returns:
        Dummy test configuration.

    """
    config = omegaconf.OmegaConf.create(
        {
            "data": {
                "data_type": "pmhc",
                "chain_lengths": {"A": 180, "C": 13},
                "threshold": 6,
            },
            "model": {
                "model_name": "TrEGNN",
                "seed": 0,
                "epochs": 1,
                "dump_pdb": True,
                "num_workers": 1,
                "batch_size": 2,
                "use_coords": False,
                "pos_encoding_type": "learned",
                "compute_dist_matrix": True,
                "coord_index": 0,
                "max_chain1_len": 180,
                "max_chain2_len": 13,
                "use_gpu": False,
            },
            "seed": {"seed": 0},
        }
    )
    return config
