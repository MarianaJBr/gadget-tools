import os

import pytest
from dotenv import load_dotenv
from sims_toolkit.gadget import File

load_dotenv()


@pytest.fixture
def snapshot_path():
    """Return the path of a testing snapshot file."""
    alt_snap_format = bool(os.getenv("ST_ALT_SNAP_FORMAT"))
    if alt_snap_format:
        path_env_var = os.getenv("ST_TEST_SNAPSHOT")
    else:
        path_env_var = os.getenv("ST_TEST_SNAPSHOT_NO_ALT")
    snapshot_path = os.path.abspath(os.path.expanduser(path_env_var))
    if not os.path.exists(snapshot_path):
        pytest.skip("snapshot file not available")
    return snapshot_path


def test_read(snapshot_path):
    """"""
    with File(snapshot_path) as snapshot:
        print(snapshot)


def test_read_invalid_mode(snapshot_path):
    """"""
    with pytest.raises(ValueError):
        # Although snapshots are open as binary, the mode does not accept
        # the ``b`` modifier.
        with File(snapshot_path, "rb") as snapshot:
            print(snapshot)


def test_inspect(snapshot_path):
    """"""
    with File(snapshot_path) as snapshot:
        block_specs = snapshot.inspect()
        print(block_specs)


def test_load_blocks(snapshot_path):
    """"""
    with File(snapshot_path) as snapshot:
        positions1 = snapshot["POS"]
        velocities = snapshot["ID"]
        positions2 = snapshot["POS"]
        print(snapshot.header)
        print(positions1)
        print(positions2)
        print(velocities)


def test_iter(snapshot_path):
    """"""
    with File(snapshot_path) as snapshot:
        for block in snapshot:
            print("************ Block ************")
            print(block)

        print("Number of reachable snapshot blocks: {}".format(len(snapshot)))
