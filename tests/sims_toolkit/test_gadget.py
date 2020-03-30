import os

import attr
import pytest
from dotenv import load_dotenv
from sims_toolkit.gadget import load_snapshot

load_dotenv()


def test_load_snapshot():
    """"""
    snapshot = os.path.expanduser(os.getenv("ST_TEST_SNAPSHOT"))
    if not os.path.exists(snapshot):
        pytest.skip("snapshot file not available")
    with open(snapshot, "rb") as g2fp:
        snapshot_data = load_snapshot(g2fp)
    print(snapshot_data)


def test_load_header():
    """"""
    snapshot = os.path.expanduser(os.getenv("ST_TEST_SNAPSHOT"))
    if not os.path.exists(snapshot):
        pytest.skip("snapshot file not available")
    with open(snapshot, "rb") as g2fp:
        snapshot_data_dict = load_snapshot(g2fp, blocks=())
    snapshot_data_dict = attr.asdict(snapshot_data_dict, recurse=False)
    header = snapshot_data_dict.pop("header")
    assert header is not None
    for id_, block in snapshot_data_dict.items():
        assert block is None
