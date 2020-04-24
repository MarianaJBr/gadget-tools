import os

import pytest
from dotenv import load_dotenv
from sims_toolkit.gadget import File
from sims_toolkit.gadget.snapshot import FileFormat

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
        for block in snapshot.values():
            print("************ Block ************")
            print(block)
        print("Number of reachable snapshot blocks: {}".format(len(snapshot)))


def test_copy_snapshot(snapshot_path):
    """"""
    with File(snapshot_path) as snap:
        new_snap_path = snapshot_path + "-test-snap"
        with File(new_snap_path, "w") as new_snap:
            # First assign the header.
            new_snap.header = snap.header
            # Block assignment order matters.
            new_snap["POS"] = snap["POS"]
            new_snap["VEL"] = snap["VEL"]
            new_snap["ID"] = snap["ID"]


def test_copy(snapshot_path):
    with File(snapshot_path) as snap:
        new_snap_path = snapshot_path + "-test-snap"
        with File(new_snap_path) as new_snap:
            assert new_snap.format is FileFormat.ALT
            assert new_snap.header == snap.header
            print(new_snap.inspect())
