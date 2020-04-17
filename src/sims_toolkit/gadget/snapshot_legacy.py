import io
import pathlib
import sys
import typing as typ
from abc import ABCMeta, abstractmethod
from enum import Enum, unique
from itertools import accumulate, starmap

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


def skip(file: BinaryIO_T, size: int, reverse: bool = False):
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
class NumParSpec:
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
class MassSpec:
    gas: float
    halo: float
    disk: float
    bulge: float
    starts: float
    bndry: float


@attr.s(auto_attribs=True)
class Header:
    """Snapshot Header."""
    num_par_spec: NumParSpec
    mass_spec: MassSpec
    time: float
    redshift: float
    flag_sfr: int
    flag_feedback: int
    num_par_total: NumParSpec
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
        skip(file, size=size - data.nbytes)
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
        assert data.dtype == header_dtype
        num_par_spec = NumParSpec(*data['Npart'])
        mass_spec = MassSpec(*data["Massarr"])
        num_par_total = NumParSpec(*data["Nall"])
        header = cls(num_par_spec=num_par_spec,
                     mass_spec=mass_spec,
                     time=data["Time"],
                     redshift=data["Redshift"],
                     flag_sfr=data["FlagSfr"],
                     flag_feedback=data["FlagFeedback"],
                     num_par_total=num_par_total,
                     flag_cooling=data["FlagCooling"],
                     num_files_snap=data["NumFiles"],
                     box_size=data["BoxSize"],
                     omega_zero=data["Omega0"],
                     omega_lambda=data["OmegaLambda"],
                     hubble_param=data["HubbleParam"])
        return header


class Block(metaclass=ABCMeta):
    """Snapshot data block."""

    @staticmethod
    @abstractmethod
    def data_from_file(file: BinaryIO_T, header: Header):
        """Read the block data from file."""
        pass

    @classmethod
    @abstractmethod
    def from_data(cls, data: np.ndarray, header: Header):
        """Creates a Block instance from a ``numpy`` array."""
        pass


def split_block_data(data: np.ndarray, num_par_spec_dict: typ.Dict[str, int]):
    """Split a block data accordingly to the number of particles
    spec in a snapshot.

    :param data: Data ``numpy`` array.
    :param num_par_spec_dict: The number of particles spec .
    :return: The split data accordingly to the particle type.
    """
    par_types_names = num_par_spec_dict.keys()
    num_par_per_type = list(num_par_spec_dict.values())
    split_idxs = list(accumulate(num_par_per_type))[:-1]
    per_type_data: typ.List[np.ndarray] = np.array_split(data, split_idxs)
    par_types_name_data = zip(par_types_names, per_type_data)

    def adjust_data(par_name: str, par_data: np.ndarray):
        """"""
        return par_name, None if not par_data.size else par_data

    return dict(starmap(adjust_data, par_types_name_data))


# noinspection DuplicatedCode
@attr.s(auto_attribs=True)
class Positions(Block):
    """Positions of the particles."""

    gas: typ.Optional[np.ndarray] = None
    halo: typ.Optional[np.ndarray] = None
    disk: typ.Optional[np.ndarray] = None
    bulge: typ.Optional[np.ndarray] = None
    stars: typ.Optional[np.ndarray] = None
    bndry: typ.Optional[np.ndarray] = None

    @staticmethod
    def data_from_file(file: BinaryIO_T, header: Header):
        """Read the positions data from file.

        :param file: Snapshot file.
        :param header: The snapshot header.
        :return: The positions data as ``numpy`` array.
        """
        size = read_size_from_delim(file)
        num_par_total = header.num_par_spec.total
        num_items = num_par_total * 3
        data: np.ndarray = np.fromfile(file, dtype="f4", count=num_items)
        skip_block_delim(file)
        assert size == data.nbytes
        return data.reshape((num_par_total, 3))

    @classmethod
    def from_data(cls, data: np.ndarray, header: Header):
        """Read the positions block from file.

        :param data: Data ``numpy`` array.
        :param header: The snapshot header.
        :return: The positions data as a ``Position`` type instance.
        """
        num_par_spec = header.num_par_spec
        num_par_spec_dict = attr.asdict(num_par_spec)
        par_pos_data = split_block_data(data, num_par_spec_dict)
        return cls(**par_pos_data)


# noinspection DuplicatedCode
@attr.s(auto_attribs=True)
class Velocities(Block):
    """Velocities of the particles."""

    gas: typ.Optional[np.ndarray] = None
    halo: typ.Optional[np.ndarray] = None
    disk: typ.Optional[np.ndarray] = None
    bulge: typ.Optional[np.ndarray] = None
    stars: typ.Optional[np.ndarray] = None
    bndry: typ.Optional[np.ndarray] = None

    @staticmethod
    def data_from_file(file: BinaryIO_T, header: Header):
        """Read the velocities data from a snapshot file.

        :param file: Snapshot file.
        :param header: The snapshot header.
        :return: The velocities data as a ```numpy`` array.
        """
        return Positions.data_from_file(file, header)

    @classmethod
    def from_data(cls, data: np.ndarray, header: Header):
        """Read the velocities block from file.

        :param data: Data ``numpy`` array.
        :param header: The snapshot header.
        :return: The velocities data as a ``Velocity`` type instance.
        """
        num_par_spec = header.num_par_spec
        num_par_spec_dict = attr.asdict(num_par_spec)
        par_pos_data = split_block_data(data, num_par_spec_dict)
        return cls(**par_pos_data)


@attr.s(auto_attribs=True)
class IDs(Block):
    """Particles identifiers."""

    gas: typ.Optional[np.ndarray] = None
    halo: typ.Optional[np.ndarray] = None
    disk: typ.Optional[np.ndarray] = None
    bulge: typ.Optional[np.ndarray] = None
    stars: typ.Optional[np.ndarray] = None
    bndry: typ.Optional[np.ndarray] = None

    @staticmethod
    def data_from_file(file: BinaryIO_T, header: Header):
        """Read the particles ids data from a snapshot file.

        :param file: Snapshot file.
        :param header: The snapshot header.
        :return: The identifiers as a ``IDs`` type instance.
        """
        size = read_size_from_delim(file)
        num_items = header.num_par_spec.total
        data: np.ndarray = np.fromfile(file, dtype="i4", count=num_items)
        skip_block_delim(file)
        assert size == data.nbytes
        return data

    @classmethod
    def from_data(cls, data: np.ndarray, header: Header):
        """Read the particles ids block from file.

        :param data: Data ``numpy`` array.
        :param header: The snapshot header.
        :return: The particles ids data as a ``IDs`` type instance.
        """
        num_par_spec = header.num_par_spec
        num_par_spec_dict = attr.asdict(num_par_spec)
        par_pos_data = split_block_data(data, num_par_spec_dict)
        return cls(**par_pos_data)


@attr.s(auto_attribs=True)
class SnapshotData:
    """Snapshot File Data"""
    header: Header
    positions: typ.Optional[np.ndarray] = None
    velocities: typ.Optional[np.ndarray] = None
    ids: typ.Optional[np.ndarray] = None
    masses: typ.Optional[np.ndarray] = None


@attr.s(auto_attribs=True)
class Snapshot:
    """"""
    path: pathlib.Path
    header: Header
    positions: typ.Optional[Positions] = None
    velocities: typ.Optional[Velocities] = None
    ids: typ.Optional[IDs] = None


@attr.s(auto_attribs=True)
class BlockIDValue:
    py_id: str
    type: typ.Type[Block] = None


@unique
class BlockID(Enum):
    """"""
    HEAD = BlockIDValue("header")
    POS = BlockIDValue("positions", Positions)
    VEL = BlockIDValue("velocities", Velocities)
    ID = BlockIDValue("ids", IDs)
    MASS = BlockIDValue("masses")
    U = BlockIDValue("internal_energy")
    RHO = BlockIDValue("density")
    HSML = BlockIDValue("smoothing_length")
    POT = BlockIDValue("potential")
    ACCE = BlockIDValue("acceleration")
    ENDT = BlockIDValue("entropy_rate_of_change")
    TSTP = BlockIDValue("time_step")

    def __init__(self, value: BlockIDValue):
        self.py_id = value.py_id
        self.type = value.type

    @staticmethod
    def all():
        """Return all available BlockID members."""
        return list(BlockID.__members__.values())

    @staticmethod
    def common():
        """Return most common BlockID members."""
        return [BlockID.POS, BlockID.VEL, BlockID.ID]


@attr.s(auto_attribs=True)
class BlockSpec:
    """"""
    total_size: int
    data_stream_pos: int
    id_str: str = None

    @property
    def id(self) -> BlockID:
        """"""
        return BlockID[self.id_str]

    def seek_stream_pos(self, file: BinaryIO_T):
        """"""
        file.seek(self.data_stream_pos)


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
        data_stream_pos = file.tell()
    else:
        # This is the data block. There is no additional block to read
        # the data block ID from.
        id_str = None
        total_size = size + 2 * BLOCK_DELIM_SIZE
        # Return to the start of the data block.
        skip_block_delim(file, reverse=True)
        data_stream_pos = file.tell()
    return BlockSpec(total_size, data_stream_pos, id_str)


def inspect_struct(file: BinaryIO_T,
                   alt_snap_format: bool = True):
    """Inspect the basic structure of a snapshot.

    :param file: A snapshot file object opened in binary mode.
    :param alt_snap_format: If ``True``, the routine assumes that the
        snapshot was created with ``SnapFormat=2``.
    """
    # Read snapshot header spec.
    header_spec = read_block_spec(file, alt_snap_format)
    # Update the ID string.
    header_spec = attr.evolve(header_spec, id_str=BlockID.HEAD.name)
    yield header_spec
    skip(file, header_spec.total_size)
    # Read the rest of the blocks.
    try:
        while True:
            block_spec = read_block_spec(file, alt_snap_format)
            yield block_spec
            skip(file, block_spec.total_size)
    except SnapshotEOFError:
        # Iteration has been broken as expected. Just continue
        # with the code execution.
        return


T_BlockSpecIter = typ.Iterator[BlockSpec]


def load_blocks_specs(file: BinaryIO_T,
                      blocks: typ.Sequence[BlockID] = None,
                      alt_snap_format: bool = True) -> T_BlockSpecIter:
    """Load the requested blocks specs from a snapshot file.

    :param file: A snapshot file object opened in binary mode.
    :param blocks: The blocks to load from the snapshot. If this argument
        is ``None`` or an empty sequence, the routine only loads the header.
    :param alt_snap_format: If ``True``, the routine assumes that the
        snapshot was created with ``SnapFormat=2``.
    :return: The snapshot blocks specs.
    """
    blocks_ids: typ.List[BlockID] = list(blocks or [])
    if BlockID.HEAD in blocks_ids:
        blocks_ids.remove(BlockID.HEAD)
    blocks_ids.insert(0, BlockID.HEAD)

    def should_load(_block_spec: BlockSpec):
        """Should we load a particular block?"""
        try:
            if _block_spec.id in blocks_ids:
                return True
            if _block_spec.id.type is None:
                return False
        except KeyError:
            pass
        return False

    def patch_spec(_block_spec: BlockSpec, block_id: BlockID):
        """Set the ID string of a BlockSpec manually."""
        id_str = block_id.name
        return attr.evolve(_block_spec, id_str=id_str)

    # Get the concrete structure, not a lazy iterator.
    snapshot_struct = list(inspect_struct(file, alt_snap_format))
    if alt_snap_format:
        blocks_specs = filter(should_load, snapshot_struct)
    else:
        blocks_specs_ids = zip(snapshot_struct, blocks_ids)
        blocks_specs = starmap(patch_spec, blocks_specs_ids)
    return blocks_specs


def load_snapshot_data(file: BinaryIO_T,
                       blocks: typ.Sequence[BlockID] = None,
                       alt_snap_format: bool = True):
    """Load the data from a snapshot file.

    :param file: A snapshot file object opened in binary mode.
    :param blocks: The blocks to load from the snapshot. If this argument
        is ``None`` or an empty sequence, the routine only loads the header.
    :param alt_snap_format: If ``True``, the routine assumes that the
        snapshot was created with ``SnapFormat=2``.
    :return: The snapshot data.
    """
    # Load the requested blocks specs.
    blocks_specs = load_blocks_specs(file, blocks, alt_snap_format)
    # Read snapshot header.
    header_spec = next(blocks_specs)
    header_spec.seek_stream_pos(file)
    header = Header.from_file(file)
    snapshot_spec = {
        header_spec.id.py_id: header
    }
    # Read the rest of the blocks.
    for block_spec in blocks_specs:
        block_spec.seek_stream_pos(file)
        block_type = block_spec.id.type
        block_py_id = block_spec.id.py_id
        block_data = block_type.data_from_file(file, header)
        snapshot_spec[block_py_id] = block_data
    return SnapshotData(**snapshot_spec)
