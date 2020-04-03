import os
import pathlib

import attr
import pytest
from dotenv import load_dotenv
from sims_toolkit.gadget import load_snapshot

load_dotenv()


@pytest.fixture
def snapshot_path():
    """Return the path of a testing snapshot file."""
    path_env_var = os.getenv("ST_TEST_SNAPSHOT")
    snapshot_path = os.path.abspath(os.path.expanduser(path_env_var))
    if not os.path.exists(snapshot_path):
        pytest.skip("snapshot file not available")
    return pathlib.Path(snapshot_path)


@pytest.fixture
def alt_snapshot_path():
    """Return the path of a testing snapshot file."""
    path_env_var = os.getenv("ST_TEST_SNAPSHOT_NO_ALT")
    snapshot_path = os.path.abspath(os.path.expanduser(path_env_var))
    if not os.path.exists(snapshot_path):
        pytest.skip("snapshot file not available")
    return pathlib.Path(snapshot_path)


def test_load_snapshot(snapshot_path):
    """"""
    with open(snapshot_path, "rb") as g2fp:
        snapshot_data = load_snapshot(g2fp)
    print(snapshot_data)


def test_load_snapshot_no_alt(alt_snapshot_path):
    """"""
    with open(alt_snapshot_path, "rb") as g2fp:
        snapshot_data = load_snapshot(g2fp, blocks=(),
                                      alt_snap_format=False)
    print(snapshot_data)


def test_load_header(snapshot_path):
    """"""
    with open(snapshot_path, "rb") as g2fp:
        snapshot_data = load_snapshot(g2fp, blocks=())
    snapshot_data_dict = attr.asdict(snapshot_data, recurse=False)
    header = snapshot_data_dict.pop("header")
    assert header is not None
    for id_, block in snapshot_data_dict.items():
        assert block is None
