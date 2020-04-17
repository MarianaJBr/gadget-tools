import os

import attr
import pytest
from dotenv import load_dotenv
from sims_toolkit.gadget import load_snapshot_data
from sims_toolkit.gadget.snapshot_legacy import (
    BlockID, Header, inspect_struct
)

load_dotenv()


@attr.s(auto_attribs=True)
class SnapshotSpec:
    """"""
    path: str
    alt_snap_format: bool = True


@pytest.fixture
def snapshot_spec():
    """Return the path of a testing snapshot file."""
    alt_snap_format = bool(os.getenv("ST_ALT_SNAP_FORMAT"))
    if alt_snap_format:
        path_env_var = os.getenv("ST_TEST_SNAPSHOT")
    else:
        path_env_var = os.getenv("ST_TEST_SNAPSHOT_NO_ALT")
    snapshot_path = os.path.abspath(os.path.expanduser(path_env_var))
    if not os.path.exists(snapshot_path):
        pytest.skip("snapshot file not available")
    return SnapshotSpec(snapshot_path, alt_snap_format)


def test_inspect_struct(snapshot_spec):
    """"""
    path = snapshot_spec.path
    alt_snap_format = snapshot_spec.alt_snap_format
    with open(path, "rb") as g2fp:
        snap_struct = inspect_struct(g2fp, alt_snap_format)
        for blocks_spec in snap_struct:
            print(blocks_spec)


def test_load_snapshot_data(snapshot_spec):
    """"""
    path = snapshot_spec.path
    alt_snap_format = snapshot_spec.alt_snap_format
    with open(path, "rb") as g2fp:
        snapshot_data = load_snapshot_data(g2fp, blocks=BlockID.common(),
                                           alt_snap_format=alt_snap_format)
    print(snapshot_data)


def test_load_header(snapshot_spec):
    """"""
    path = snapshot_spec.path
    alt_snap_format = snapshot_spec.alt_snap_format
    with open(path, "rb") as g2fp:
        snapshot_data = load_snapshot_data(g2fp, blocks=(),
                                           alt_snap_format=alt_snap_format)
    snapshot_data_dict = attr.asdict(snapshot_data, recurse=False)
    header = snapshot_data_dict.pop("header")
    assert header is not None
    for id_, block in snapshot_data_dict.items():
        assert block is None


def test_header_as_data(snapshot_spec):
    """"""
    path = snapshot_spec.path
    alt_snap_format = snapshot_spec.alt_snap_format
    with open(path, "rb") as g2fp:
        snapshot_data = load_snapshot_data(g2fp, blocks=(),
                                           alt_snap_format=alt_snap_format)
    header = snapshot_data.header
    header_as_data = header.as_data()
    assert header == Header.from_data(header_as_data)
