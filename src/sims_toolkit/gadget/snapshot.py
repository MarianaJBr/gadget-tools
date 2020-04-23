import io
import os
import sys
import typing as t
from collections.abc import Mapping
from contextlib import AbstractContextManager
from enum import Enum, unique
from itertools import accumulate, starmap

import attr
import numpy as np

# Size of delimiter blocks.
SNAP_START_POS = 0
BLOCK_DELIM_SIZE = 4

# Size of the data chunk where the block id is stored in snapshots
# with SnapFormat=2.
ID_CHUNK_SIZE = 4

# Size of the identifier block in snapshots with SnapFormat=2.
ALT_ID_BLOCK_SIZE = 2 * ID_CHUNK_SIZE

# Size in bytes of the header.
HEADER_SIZE = 256

# Typing helpers.
T_BinaryIO = t.BinaryIO

# Valid file modes for handling snapshots.
FILE_MODES = frozenset({"r", "w", "x", "a"})

# Some useful attrs filters.
EXCLUDE_NONE_FILTER = attr.filters.exclude(type(None))


@unique
class FileFormat(Enum):
    """Snapshot formats"""
    DEFAULT = 1
    ALT = 2


class FormatWarning(RuntimeWarning):
    """A snapshot with ``FileFormat.DEFAULT`` has not a fully defined
    struct."""
    pass


class SnapshotEOFError(EOFError):
    """Read beyond end of GADGET-2 snapshot file."""
    pass


class FormatError(ValueError):
    """Real and expected structure of Snapshot do not match."""
    pass


def read_size_from_delim(file: T_BinaryIO):
    """Read a delimiter block and return its contents. The returned value
    is the size in bytes of the following data block.

    :param file: Snapshot file.
    :return: Size of following data block in bytes.
    """
    size_bytes = file.read(BLOCK_DELIM_SIZE)
    if size_bytes == b"":
        raise SnapshotEOFError
    return int.from_bytes(size_bytes, sys.byteorder)


def skip_block_delim(file: T_BinaryIO, reverse: bool = False):
    """Skip a delimiter block.

    :param file: Snapshot file.
    :param reverse: Skip the block backwards.
    """
    size = -BLOCK_DELIM_SIZE if reverse else BLOCK_DELIM_SIZE
    file.seek(size, io.SEEK_CUR)


def skip(file: T_BinaryIO, size: int, reverse: bool = False):
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
    def from_file(cls, file: T_BinaryIO):
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


@attr.s(auto_attribs=True)
class BlockSpec:
    """"""
    total_size: int
    data_stream_pos: int
    id: str = None


@attr.s(auto_attribs=True, frozen=True)
class BlockData:
    """The block data, organized by particle type."""
    gas: t.Optional[np.ndarray] = None
    halo: t.Optional[np.ndarray] = None
    disk: t.Optional[np.ndarray] = None
    bulge: t.Optional[np.ndarray] = None
    stars: t.Optional[np.ndarray] = None
    bndry: t.Optional[np.ndarray] = None


@attr.s(auto_attribs=True, frozen=True)
class Block:
    """A snapshot data block."""
    id: str
    data: BlockData


def split_block_data(data: np.ndarray, num_par_spec_dict: t.Dict[str, int]):
    """Split a block data accordingly to the number of particles
    spec in a snapshot.

    :param data: Data ``numpy`` array.
    :param num_par_spec_dict: The number of particles spec .
    :return: The split data accordingly to the particle type.
    """
    par_types_names = num_par_spec_dict.keys()
    num_par_per_type = list(num_par_spec_dict.values())
    split_idxs = list(accumulate(num_par_per_type))[:-1]
    per_type_data: t.List[np.ndarray] = np.array_split(data, split_idxs)
    par_types_name_data = zip(par_types_names, per_type_data)

    def adjust_data(par_name: str, par_data: np.ndarray):
        """"""
        return par_name, None if not par_data.size else par_data

    return dict(starmap(adjust_data, par_types_name_data))


# Typing helpers.
T_DataLoader = t.Callable[[T_BinaryIO, Header], t.Dict[str, np.ndarray]]
T_DataLoaders = t.Dict[str, T_DataLoader]
T_Struct = t.Dict[str, BlockSpec]


@attr.s(auto_attribs=True)
class DataLoaders:
    """Procedures used to load data from the snapshot blocks.

    This class defines the python callable objects used to load data from
    the blocks in a GADGET snapshot, according to the standard snapshot
    specification (see GADGET-2 manual, p.32).
    """
    POS: t.Type[T_DataLoader] = None
    VEL: t.Type[T_DataLoader] = None
    ID: t.Type[T_DataLoader] = None
    MASS: t.Type[T_DataLoader] = None
    U: t.Type[T_DataLoader] = None
    RHO: t.Type[T_DataLoader] = None
    HSML: t.Type[T_DataLoader] = None
    POT: t.Type[T_DataLoader] = None
    ACCE: t.Type[T_DataLoader] = None
    ENDT: t.Type[T_DataLoader] = None
    TSTP: t.Type[T_DataLoader] = None


@attr.s(auto_attribs=True, frozen=True)
class File(AbstractContextManager, Mapping):
    """Represent a GADGET-2 snapshot file."""

    name: os.PathLike
    mode: t.Optional[str] = "r"
    data_loaders: T_DataLoaders = attr.ib(default=None)
    _file: T_BinaryIO = attr.ib(default=None, init=False, repr=False)
    _format: FileFormat = attr.ib(default=None, init=False, repr=False)
    _header: Header = attr.ib(default=None, init=False, repr=False)
    _struct: T_Struct = attr.ib(default=None, init=False, repr=False)

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        mode = self.mode
        if mode not in FILE_MODES:
            raise ValueError(f"invalid mode; mode must be one of "
                             f"{FILE_MODES}")
        # Force the file to be open in binary mode.
        mode += "b"
        # noinspection PyTypeChecker
        file: T_BinaryIO = open(self.name, mode)
        object.__setattr__(self, "_file", file)
        # ************ Define the snapshot structure ************
        if not self.size:
            # Empty, writable files will have FileFormat.ALT.
            object.__setattr__(self, "_format", FileFormat.ALT)
            object.__setattr__(self, "_struct", {})
        else:
            # ************ Define the snapshot Format ************
            _format = self._detect_format()
            object.__setattr__(self, "_format", _format)
            # ************ Initialize the header ************
            header = Header.from_file(file)
            file.seek(SNAP_START_POS)
            object.__setattr__(self, "_header", header)
            # ************ Define the block data loaders **************
            if self.data_loaders is None:
                # These are the default block data loaders
                data_loaders = DataLoaders(POS=self._load_positions,
                                           VEL=self._load_velocities,
                                           ID=self._load_ids)
                data_loaders_dict: T_DataLoaders \
                    = attr.asdict(data_loaders, filter=EXCLUDE_NONE_FILTER)
                object.__setattr__(self, "data_loaders", data_loaders_dict)
            # ************ Load block specs ************
            struct: T_Struct = self._define_struct()
            object.__setattr__(self, "_struct", struct)

    @property
    def header(self):
        """The snapshot header."""
        return self._header

    @property
    def format(self):
        """Detect the snapshot format"""
        return self._format

    @property
    def size(self):
        """Size of the snapshot in bytes"""
        file_size = self._goto_end()
        self._goto_start()
        return file_size

    def _detect_format(self):
        """Detect the snapshot file format.

        :return: The snapshot format.
        """
        self__file = self._file
        size = read_size_from_delim(self__file)
        if size not in [HEADER_SIZE, ALT_ID_BLOCK_SIZE]:
            # The first block can only have two possible sizes.
            raise FormatError("this is not a valid snapshot file")
        body_bytes = self__file.read(size)
        try:
            # Try to read the block ID from the header block
            id_str_bytes = body_bytes[:ID_CHUNK_SIZE].decode("ascii")
            id_str = str(id_str_bytes).rstrip()
        except UnicodeDecodeError:
            _format = FileFormat.DEFAULT
        else:
            if id_str == "HEAD":
                _format = FileFormat.ALT
            else:
                _format = FileFormat.DEFAULT
        skip_block_delim(self__file)
        if _format is FileFormat.DEFAULT:
            # Reset stream position to the start.
            self__file.seek(SNAP_START_POS, io.SEEK_SET)
        return _format

    def _goto_start(self):
        """Go to the snapshot starting position."""
        return self._file.seek(SNAP_START_POS, io.SEEK_SET)

    def _goto_end(self):
        """Go to the snapshot ending position."""
        return self._file.seek(SNAP_START_POS, io.SEEK_END)

    def _read_size_from_delim(self):
        """Read a delimiter block and return its contents. The returned value
        is the size in bytes of the following data block.

        :return: Size of following data block in bytes.
        """
        size_bytes = self._file.read(BLOCK_DELIM_SIZE)
        if size_bytes == b"":
            raise SnapshotEOFError
        return int.from_bytes(size_bytes, sys.byteorder)

    def _read_block_spec(self):
        """Load the identified and size of a snapshot block.

        :return:
        """
        self__file = self._file
        size = self._read_size_from_delim()
        if self.format is FileFormat.ALT:
            # Read the block ID from the additional block
            body_bytes = self__file.read(size)
            id_bytes = body_bytes[:ID_CHUNK_SIZE].decode("ascii")
            _id = str(id_bytes).rstrip()
            # Get the total size (including delimiter blocks) of
            # the block's data.
            total_size_bytes = body_bytes[ID_CHUNK_SIZE:]
            total_size = int.from_bytes(total_size_bytes, sys.byteorder)
            skip_block_delim(self__file)
            data_stream_pos = self__file.tell()
        else:
            # This is the data block. There is no additional block to read
            # the data block ID from.
            _id = None
            total_size = size + 2 * BLOCK_DELIM_SIZE
            # Return to the start of the data block.
            skip_block_delim(self__file, reverse=True)
            data_stream_pos = self__file.tell()
        return BlockSpec(total_size, data_stream_pos, _id)

    def _skip_block_delim(self, reverse: bool = False):
        """Skip a delimiter block.

        :param reverse: Skip the block backwards.
        """
        size = -BLOCK_DELIM_SIZE if reverse else BLOCK_DELIM_SIZE
        self._file.seek(size, io.SEEK_CUR)

    def _skip_block(self, block_spec: BlockSpec):
        """Skip a block of ``block_spec` bytes.

        :param self: Snapshot file.
        :param block_spec: The block spec.
        """
        total_size = block_spec.total_size
        data_stream_pos = block_spec.data_stream_pos
        self._file.seek(data_stream_pos + total_size, io.SEEK_SET)

    def inspect(self):
        """Inspect the structure of a snapshot."""
        return list(self._inspect_struct())

    def _inspect_struct(self):
        """Inspect the structure of a snapshot."""
        # Read snapshot header spec.
        try:
            self._file.seek(SNAP_START_POS, io.SEEK_SET)
            header_spec = self._read_block_spec()
            # Update the ID string.
            header_spec = attr.evolve(header_spec, id="HEAD")
            yield header_spec
            self._skip_block(header_spec)
            # Read the rest of the blocks.
            while True:
                block_spec = self._read_block_spec()
                yield block_spec
                self._skip_block(block_spec)
        except SnapshotEOFError:
            # Iteration has been broken as expected. Just continue
            # with the code execution.
            return

    def _define_struct(self) -> T_Struct:
        """Define the structure of this snapshot file.

        :param self: A snapshot file object opened in binary mode.
        :return: The snapshot blocks specs.
        """
        # Exclude the header spec.
        spec_list = list(self._inspect_struct())[1:]
        loader_ids = list(self.data_loaders.keys())

        def patch_spec(spec: BlockSpec, type_id: str):
            """Set the ID string of a BlockSpec manually."""
            return attr.evolve(spec, id=type_id)

        if self.format is FileFormat.DEFAULT:
            specs_and_loader_ids = zip(spec_list, loader_ids)
            spec_list = starmap(patch_spec, specs_and_loader_ids)
            spec_map: T_Struct = dict(zip(loader_ids, spec_list))
        else:
            spec_map = {spec.id: spec for spec in spec_list}
        return {loader_id: spec_map.get(loader_id, None) for loader_id in
                loader_ids}

    def _goto_block(self, block_spec: BlockSpec):
        """Jump directly to the block data location in the snapshot.

        :param block_spec: The block spec.
        """
        data_stream_pos = block_spec.data_stream_pos
        self._file.seek(data_stream_pos, io.SEEK_SET)

    @staticmethod
    def _load_positions(file: T_BinaryIO, header: Header):
        """Load positions data from a binary file.

        :param file: Snapshot file.
        :param header: The snapshot header.
        :return: The positions data as ``numpy`` array.
        """
        num_par_spec = header.num_par_spec
        size = read_size_from_delim(file)
        num_par_total = num_par_spec.total
        num_items = num_par_total * 3
        data: np.ndarray = np.fromfile(file, dtype="f4", count=num_items)
        skip_block_delim(file)
        assert size == data.nbytes
        positions = data.reshape((num_par_total, 3))
        num_par_spec_dict = attr.asdict(num_par_spec)
        return split_block_data(positions, num_par_spec_dict)

    def _load_velocities(self, file: T_BinaryIO, header: Header):
        """Load velocities data from a binary file.

        :param file: Snapshot file.
        :param header: The snapshot header.
        :return: The velocities data as a ```numpy`` array.
        """
        return self._load_positions(file, header)

    @staticmethod
    def _load_ids(file: T_BinaryIO, header: Header):
        """Load identifiers data from a binary file.

        :param file: Snapshot file.
        :param header: The snapshot header.
        :return: The identifiers as a ``IDs`` type instance.
        """
        size = read_size_from_delim(file)
        num_par_spec = header.num_par_spec
        num_items = num_par_spec.total
        data: np.ndarray = np.fromfile(file, dtype="i4", count=num_items)
        skip_block_delim(file)
        assert size == data.nbytes
        num_par_spec_dict = attr.asdict(num_par_spec)
        return split_block_data(data, num_par_spec_dict)

    def __getitem__(self, block_id: str):
        """Return item"""
        if block_id == "HEAD":
            return self.header
        data_loader = self.data_loaders[block_id]
        block_spec = self._struct[block_id]
        if data_loader is None:
            raise TypeError("block type is not defined")
        if block_spec is None:
            raise FormatError("block not found in snapshot")
        self._goto_block(block_spec)
        block_data_dict = data_loader(self._file, self.header)
        block_data = BlockData(**block_data_dict)
        return Block(block_id, block_data)

    def __len__(self) -> int:
        return len(self.data_loaders)

    def __iter__(self):
        yield self.header
        block_ids = list(self.data_loaders.keys())
        for block_id in block_ids:
            yield self[block_id]

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
