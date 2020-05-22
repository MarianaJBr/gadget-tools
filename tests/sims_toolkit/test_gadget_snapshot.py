import os
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv
from sims_toolkit.gadget import Snapshot
from sims_toolkit.gadget.snapshot import (
    FileFormat, Header, header_dtype
)

load_dotenv()


@pytest.fixture
def snapshot_path():
    """Return the path of a testing snapshot file."""
    path_env_var = os.getenv("ST_TEST_SNAPSHOT")
    snap_path = os.path.abspath(os.path.expanduser(path_env_var))
    if not os.path.exists(snap_path):
        pytest.skip("snapshot file not available")
    return snap_path


@pytest.fixture
def dummy_snap_header():
    """"""
    # Header attributes from a real snapshot.
    header_attrs = ((2479138, 2539061, 0, 0, 99736, 0),
                    (0.0, 0.053677940511483495, 0.0, 0.0, 0.0, 0.0),
                    0.24999999949102492, 3.0000000081436013, 1, 1,
                    (125791124, 134217728, 0, 0, 8426604, 0), 1, 64,
                    100000.0, 0.308, 0.692, 0.678)
    header_data = np.array(header_attrs, dtype=header_dtype)
    return Header.from_data(header_data)


@pytest.fixture
def dummy_snap_block(dummy_snap_header):
    """

    :param dummy_snap_header:
    :return:
    """
    num_par_gas = dummy_snap_header.num_pars.gas
    num_par_halo = dummy_snap_header.num_pars.halo
    num_par_stars = dummy_snap_header.num_pars.stars
    # Data for types gas, halo, disk, bulge, stars, bndry.
    return {
        "gas": np.random.random_sample((num_par_gas, 3)).astype("f4"),
        "halo": np.random.random_sample((num_par_halo, 3)).astype("f4"),
        "stars": np.random.random_sample((num_par_stars, 3)).astype("f4"),
    }


def test_read(snapshot_path):
    """"""
    with Snapshot(snapshot_path) as snap:
        print(snap)


def test_read_invalid_mode(snapshot_path):
    """"""
    with pytest.raises(ValueError):
        # Although snapshots are open as binary, the mode does not accept
        # the ``b`` modifier.
        with Snapshot(snapshot_path, "rb") as snap:
            print(snap)


def test_inspect(snapshot_path):
    """"""
    with Snapshot(snapshot_path) as snap:
        block_specs = snap.inspect()
        print(block_specs)


def test_load_blocks(snapshot_path):
    """"""
    with Snapshot(snapshot_path) as snap:
        positions1 = snap["POS"]
        velocities = snap["VEL"]
        positions2 = snap["POS"]
        print(snap.header)
        print(positions1)
        print(positions2)
        print(velocities)


def test_iter(snapshot_path):
    """"""
    with Snapshot(snapshot_path) as snap:
        for block in snap.values():
            print("************ Block ************")
            print(block)
        print("Number of reachable snapshot blocks: {}".format(len(snap)))


def test_copy(snapshot_path):
    """"""
    with Snapshot(snapshot_path) as snap:
        new_snap_path = snapshot_path + "-test-copy"
        with Snapshot(new_snap_path, "w") as new_snap:
            # First assign the header.
            new_snap.header = snap.header
            # Block assignment order matters.
            new_snap["POS"] = snap["POS"]
            new_snap["VEL"] = snap["VEL"]
            new_snap["ID"] = snap["ID"]
            new_snap.flush()


def test_compare_copy(snapshot_path):
    """"""
    with Snapshot(snapshot_path) as snap:
        new_snap_path = snapshot_path + "-test-copy"
        with Snapshot(new_snap_path) as new_snap:
            assert new_snap.format is FileFormat.ALT
            assert new_snap.header == snap.header
            print(new_snap.inspect())


def test_copy_default_format(snapshot_path):
    """"""
    with Snapshot(snapshot_path) as snap:
        new_snap_path = snapshot_path + "-test-copy"
        with Snapshot(new_snap_path, "w",
                      format=FileFormat.DEFAULT) as new_snap:
            # First assign the header.
            new_snap.header = snap.header
            # Block assignment order matters.
            new_snap["POS"] = snap["POS"]
            new_snap["VEL"] = snap["VEL"]
            new_snap["ID"] = snap["ID"]
            new_snap.flush()


def test_compare_copy_default_format(snapshot_path):
    """"""
    with Snapshot(snapshot_path) as snap:
        new_snap_path = snapshot_path + "-test-copy"
        with Snapshot(new_snap_path) as new_snap:
            assert new_snap.format is FileFormat.DEFAULT
            assert new_snap.header == snap.header
            print(new_snap.inspect())


def test_copy_block_twice(snapshot_path):
    """"""
    with pytest.raises(KeyError):
        with Snapshot(snapshot_path) as snap:
            new_snap_path = snapshot_path + "-test-snap"
            with Snapshot(new_snap_path, "w") as new_snap:
                # First assign the header.
                new_snap.header = snap.header
                # This must fail.
                new_snap["POS"] = snap["POS"]
                new_snap.flush()
                new_snap["POS"] = snap["POS"]
                new_snap.flush()


def test_create(dummy_snap_header, dummy_snap_block):
    """"""
    file_path = Path(os.path.join(os.getcwd(), "test-dummy-snap"))
    file_format = FileFormat.ALT
    with Snapshot(file_path, "w", format=file_format) as snap:
        # Assign new header and block.
        snap.header = dummy_snap_header
        snap["POS"] = dummy_snap_block
        print(snap["POS"])
        snap.flush()


def test_read_created(dummy_snap_header):
    """"""
    file_path = Path(os.path.join(os.getcwd(), "test-dummy-snap"))
    with Snapshot(file_path) as snap:
        assert snap.header == dummy_snap_header
        print(snap["POS"])
