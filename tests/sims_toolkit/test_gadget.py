import os

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
    assert False
