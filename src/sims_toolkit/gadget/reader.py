import typing as typ
from contextlib import contextmanager

import attr
import numpy as np


class GADGETIOError(IOError):
    """Represent an error while accessing a GADGET file."""
    pass


@attr.s(auto_attribs=True)
class NumPartSpec:
    gas: int
    halo: int
    disk: int
    bulge: int
    starts: int
    bndry: int


@attr.s(auto_attribs=True)
class MassPartSpec:
    gas: float
    halo: float
    disk: float
    bulge: float
    starts: float
    bndry: float


@attr.s(auto_attribs=True)
class FileHeaderFields:
    """GADGET File attributes."""
    num_part_file: NumPartSpec
    mass_array: MassPartSpec
    time: float
    redshift: float
    flag_sfr: int
    flag_feedback: int
    num_part_total: NumPartSpec
    flag_cooling: int
    num_files_snap: int
    box_size: float
    omega_zero: float
    omega_lambda: float
    hubble_param: float


@contextmanager
def open_block(file: typ.BinaryIO, new_snap_format: bool):
    """
    Function that handles the two types of sizes in blocks of Gadget-2 styled
    files. With the option "--new-snap_format" passed in in the command line
    it chooses bewteen Standard format (size of blocks) and the variation
    described in section 6.2 of the GADGET guide (SnapFormat=2)
    :param file:
    :param new_snap_format:
    :return:
    """
    # Size of the current block
    try:
        block_size = np.fromfile(file, dtype="i", count=1)[0]
    except IndexError:
        raise GADGETIOError("GADGET EOF")
    if new_snap_format:
        # If the SnapFormat is 2 (the variant) then we have to read
        # the small block that precedes the main block.
        block_id = file.read(block_size)
        # Read extra block closing bytes.
        block_size_end = np.fromfile(file, dtype="i", count=1)[0]
        assert block_size == block_size_end
        print("Reading block with ID <<{}>>...".format(block_id[:4]))
        # Now read the opening bytes of the block.
        block_size = np.fromfile(file, dtype="i", count=1)[0]
    else:
        print("Reading block...")
    print("The size of this block is {:d} bytes".format(block_size))
    yield file, block_size
    block_size_end = np.fromfile(file, dtype="i", count=1)[0]
    assert block_size == block_size_end
    print("Completed")
