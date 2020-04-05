import io
import sys
import typing as typ
from abc import ABCMeta, abstractmethod
from enum import Enum, unique
from itertools import count

import attr
import numpy as np

# Size of delimiter blocks
BLOCK_DELIM_SIZE = 4
# Size of the data chunk where the block size is stored in.
BLOCK_ID_SIZE = 4

BinaryIO_T = typ.BinaryIO


class SnapshotEOFError(EOFError):
    """Read beyond end of GADGET-2 snapshot file."""
    pass


def read_size_from_delim(file: BinaryIO_T):
    """Read a delimiter block and return its contents. The returned value
    is the size in bytes of the following data block.

    :param file: Snapshot file.
    :return: Size of following data block in bytes.
    """
    size_bytes = file.read(BLOCK_DELIM_SIZE)
    if size_bytes == b"":
        raise SnapshotEOFError
    return int.from_bytes(size_bytes, sys.byteorder)


def skip_block_delim(file: BinaryIO_T, reverse: bool = False):
    """Skip a delimiter block.

    :param file: Snapshot file.
    :param reverse: Skip the block backwards.
    """
    size = -BLOCK_DELIM_SIZE if reverse else BLOCK_DELIM_SIZE
    file.seek(size, io.SEEK_CUR)


def skip_block(file: BinaryIO_T, size: int, reverse: bool = False):
    """Skip a block of ``size`` bytes.

    :param file: Snapshot file.
    :param size: Size of block in bytes.
    :param reverse: Skip the block backwards.
    """
    size = -size if reverse else size
    file.seek(size, io.SEEK_CUR)


# Numpy dtype for the snapshot header data.
# We name each field according to GADGET-2 manual.
header_dtype = np.dtype([
    ("Npart", "u4", 6),
    ("Massarr", "f8", 6),
    ("Time", "f8"),
    ("Redshift", "f8"),
    ("FlagSfr", "i4"),
    ("FlagFeedback", "i4"),
    ("Nall", "i4", 6),
    ("FlagCooling", "i4"),
    ("NumFiles", "i4"),
    ("BoxSize", "f8"),
    ("Omega0", "f8"),
    ("OmegaLambda", "f8"),
    ("HubbleParam", "f8")
])


# Size of the header in bytes
@attr.s(auto_attribs=True)
class NumPartSpec:
    gas: int
    halo: int
    disk: int
    bulge: int
    stars: int
    bndry: int

    @property
    def total(self):
        return sum(attr.astuple(self))


@attr.s(auto_attribs=True)
class MassPartSpec:
    gas: float
    halo: float
    disk: float
    bulge: float
    starts: float
    bndry: float


@attr.s(auto_attribs=True)
class Header:
    """Snapshot Header."""
    num_part: NumPartSpec
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

    @classmethod
    def from_file(cls, file: BinaryIO_T):
        """Read the snapshot file header.

        :param file: Snapshot file.
        :return: The snapshot header data as a ``Header`` type instance.
        """
        size = read_size_from_delim(file)
        data = np.fromfile(file, dtype=header_dtype, count=1)[0]
        # Skip the remaining header bytes.
        skip_block(file, size=size - data.nbytes)
        skip_block_delim(file)
        return cls.from_data(data)

    def as_data(self):
        """Convert the snapshot file header to a numpy array.

        :return: The snapshot header data as a numpy array.
        """
        data = attr.astuple(self)
        return np.array(data, dtype=header_dtype)

    @classmethod
    def from_data(cls, data: np.ndarray):
        """Read the snapshot header from a numpy array.

        :param data: The snapshot header data.
        :return: The snapshot header data as a ``Header`` type instance.
        """
        num_part_spec = NumPartSpec(*data['Npart'])
        mass_part_spec = MassPartSpec(*data["Massarr"])
        num_part_total = NumPartSpec(*data["Nall"])
        header = cls(num_part=num_part_spec,
                     mass_array=mass_part_spec,
                     time=data["Time"],
                     redshift=data["Redshift"],
                     flag_sfr=data["FlagSfr"],
                     flag_feedback=data["FlagFeedback"],
                     num_part_total=num_part_total,
                     flag_cooling=data["FlagCooling"],
                     num_files_snap=data["NumFiles"],
                     box_size=data["BoxSize"],
                     omega_zero=data["Omega0"],
                     omega_lambda=data["OmegaLambda"],
                     hubble_param=data["HubbleParam"])
        return header


class Block(metaclass=ABCMeta):
    """Snapshot data block."""

    data: np.ndarray

    @staticmethod
    @abstractmethod
    def data_from_file(file: BinaryIO_T, header: Header):
        """Read the block data from file."""
        pass


@attr.s(auto_attribs=True)
class Position(Block):
    """Positions of the particles."""

    data: np.ndarray

    @property
    def x_coord(self):
        return self.data[:, 0]

    @property
    def y_coord(self):
        return self.data[:, 1]

    @property
    def z_coord(self):
        return self.data[:, 2]

    @staticmethod
    def data_from_file(file: BinaryIO_T, header: Header):
        """Read the positions data from file.

        :param file: Snapshot file.
        :param header: The snapshot header.
        :return: The positions data as a ``Position`` type instance.
        """
        skip_block_delim(file)
        num_items = header.num_part.total * 3
        data = np.fromfile(file, dtype="f4", count=num_items)
        skip_block_delim(file)
        return data


@attr.s(auto_attribs=True)
class Velocity(Block):
    """Velocities of the particles."""

    data: np.ndarray

    @property
    def x_coord(self):
        return self.data[:, 0]

    @property
    def y_coord(self):
        return self.data[:, 1]

    @property
    def z_coord(self):
        return self.data[:, 2]

    @staticmethod
    def data_from_file(file: BinaryIO_T, header: Header):
        """Read the velocities data from a snapshot file.

        :param file: Snapshot file.
        :param header: The snapshot header.
        :return: The velocities data as a ``Velocity`` type instance.
        """
        skip_block_delim(file)
        num_items = header.num_part.total * 3
        data = np.fromfile(file, dtype="f4", count=num_items)
        skip_block_delim(file)
        return data


@attr.s(auto_attribs=True)
class IDs(Block):
    """Particles identifiers."""

    data: np.ndarray

    @staticmethod
    def data_from_file(file: BinaryIO_T, header: Header):
        """Read the particles identifiers from a snapshot file.

        :param file: Snapshot file.
        :param header: The snapshot header.
        :return: The identifiers as a ``IDs`` type instance.
        """
        skip_block_delim(file)
        num_items = header.num_part.total
        data = np.fromfile(file, dtype="i4", count=num_items)
        skip_block_delim(file)
        return data


@attr.s(auto_attribs=True)
class SnapshotData:
    """Snapshot File Data"""
    header: Header
    positions: typ.Optional[np.ndarray] = None
    velocities: typ.Optional[np.ndarray] = None
    ids: typ.Optional[np.ndarray] = None
    masses: typ.Optional[np.ndarray] = None


@unique
class BlockID(Enum):
    """"""
    HEAD = "header"
    POS = "positions"
    VEL = "velocities"
    ID = "ids"
    MASS = "masses"
    U = "internal_energy"
    RHO = "density"
    HSLM = "smoothing_length"
    POT = "potential"
    ACCE = "acceleration"
    ENDT = "entropy_rate_of_change"
    TSTP = "time_step"


@unique
class BlockType(Enum):
    """The available block types in a snapshot file (excluding the header)"""
    POS = Position
    VEL = Velocity
    ID = IDs
    # MASS = Block
    # U = Block
    # RHO = Block
    # HSLM = Block
    # POT = Block
    # ACCE = Block
    # ENDT = Block
    # TSTP = Block


@attr.s(auto_attribs=True)
class BlockSpec:
    """"""
    id_str: str
    total_size: int


def read_block_spec(file: BinaryIO_T, alt_snap_format: bool = True):
    """Load the identified and size of a snapshot block.

    :param file: A snapshot file object opened in binary mode.
    :param alt_snap_format: If ``True``, the routine assumes that the
        snapshot was created with ``SnapFormat=2``.
    :return:
    """
    size = read_size_from_delim(file)
    if alt_snap_format:
        # Read the block ID from the additional block
        body_bytes = file.read(size)
        id_bytes = body_bytes[:BLOCK_ID_SIZE].decode("ascii")
        id_str = str(id_bytes).rstrip()
        # Get the total size (including delimiter blocks) of the block's data
        total_size_bytes = body_bytes[BLOCK_ID_SIZE:]
        total_size = int.from_bytes(total_size_bytes, sys.byteorder)
        skip_block_delim(file)
    else:
        # This is the data block. There is no additional block to read
        # the data block ID from.
        id_str = "UNKNOWN"
        total_size = size + 2 * BLOCK_DELIM_SIZE
        # Return to the start of the data block.
        skip_block_delim(file, reverse=True)
    return BlockSpec(id_str, total_size)


def load_snapshot(file: BinaryIO_T,
                  blocks: typ.Sequence[BlockID] = None,
                  alt_snap_format: bool = True):
    """Load the data from a snapshot file.

    :param file: A snapshot file object opened in binary mode.
    :param blocks: The blocks to load from the snapshot. If this argument
        is ``None``, then the routine loads the whole snapshot file.
    :param alt_snap_format: If ``True``, the routine assumes that the
        snapshot was created with ``SnapFormat=2``.
    :return: The snapshot data.
    """
    block_type_members: typ.Dict[str, BlockType] = BlockType.__members__
    if blocks is None:
        # Read all of the blocks.
        blocks_ids: typ.List[BlockID] = list(BlockID.__members__.values())
        if not alt_snap_format:
            blocks_ids = []
    else:
        blocks_ids: typ.List[BlockID] = list(blocks)
        if BlockID.HEAD in blocks_ids:
            blocks_ids.remove(BlockID.HEAD)
        # blocks_ids.insert(0, BlockID.HEAD)
    snap_data = {}
    # Read snapshot header.
    header_spec = read_block_spec(file, alt_snap_format)
    if not alt_snap_format:
        header_spec = attr.evolve(header_spec, id_str=BlockID.HEAD.name)
    header_id_str = header_spec.id_str
    header = Header.from_file(file)
    # NOTE: This seems a bit redundant.
    snap_data[BlockID[header_id_str].value] = header
    # Read the rest of the blocks.
    try:
        for block_idx in count():
            if not alt_snap_format:
                if not blocks_ids[block_idx:]:
                    break
            block_spec = read_block_spec(file, alt_snap_format)
            if not alt_snap_format:
                # Set explicitly the block spec ID.
                block_id_str = blocks_ids[block_idx].name
                block_spec = attr.evolve(block_spec, id_str=block_id_str)
            block_id_str = block_spec.id_str
            if block_id_str not in block_type_members.keys():
                # Unrecognized block. Do not load any data.
                skip_block(file, block_spec.total_size)
                continue
            if BlockID[block_id_str] not in blocks_ids:
                # Block not required. Do not load any data.
                skip_block(file, block_spec.total_size)
                continue
            block_type: Block = block_type_members[block_id_str].value
            block_fancy_name = BlockID[block_id_str].value
            block_data = block_type.data_from_file(file, header)
            snap_data[block_fancy_name] = block_data
    except SnapshotEOFError:
        # Iteration has been broken as expected. Just continue
        # with the code execution.
        pass
    return SnapshotData(**snap_data)
