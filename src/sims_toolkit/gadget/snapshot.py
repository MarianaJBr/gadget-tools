import io
import os
import sys
import typing as t
from collections.abc import MutableMapping
from contextlib import AbstractContextManager
from enum import Enum, unique
from itertools import accumulate, chain, starmap

import attr
import numpy as np

# Size of delimiter blocks.
SNAP_START_POS = 0
BLOCK_DELIM_SIZE = 4

# Size of the data chunk where the block total size is stored in snapshots
# with SnapFormat=2.
SIZE_CHUNK_SIZE = 4
# Size of the data chunk where the block id is stored in snapshots
# with SnapFormat=2.
ID_CHUNK_SIZE = 4
# Size of the identifier block in snapshots with SnapFormat=2.
ALT_ID_BLOCK_SIZE = SIZE_CHUNK_SIZE + ID_CHUNK_SIZE

# Size in bytes of the header.
HEADER_SIZE = 256

# Typing helpers.
T_BinaryIO = t.BinaryIO
T_DataLoader = t.Callable[[T_BinaryIO, "Header"], t.Dict[str, np.ndarray]]
T_DataLoaders = t.Dict[str, T_DataLoader]
T_Struct = t.Dict[str, t.Union["BlockSpec", None]]
T_TempStorage = t.Dict[str, "Block"]

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


def write_block_delim(file: T_BinaryIO, block_size: int):
    """Write a delimiter block. The delimiter content is ``block_size``
    variable in bytes.

    :param file: Snapshot file.
    :param block_size:
    """
    size_bytes = int.to_bytes(block_size, BLOCK_DELIM_SIZE, sys.byteorder)
    file.write(size_bytes)


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

    @property
    def size(self):
        """"""
        return header_dtype.itemsize

    @classmethod
    def from_file(cls, file: T_BinaryIO):
        """Read the snapshot file header.

        :param file: Snapshot file.
        :return: The snapshot header data as a ``Header`` type instance.
        """
        data = np.fromfile(file, dtype=header_dtype, count=1)[0]
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
                     time=float(data["Time"]),
                     redshift=float(data["Redshift"]),
                     flag_sfr=bool(data["FlagSfr"]),
                     flag_feedback=bool(data["FlagFeedback"]),
                     num_par_total=num_par_total,
                     flag_cooling=bool(data["FlagCooling"]),
                     num_files_snap=int(data["NumFiles"]),
                     box_size=float(data["BoxSize"]),
                     omega_zero=float(data["Omega0"]),
                     omega_lambda=float(data["OmegaLambda"]),
                     hubble_param=float(data["HubbleParam"]))
        return header

    def to_file(self, file: T_BinaryIO):
        """

        :param file:
        :return:
        """
        self_attrs = attr.astuple(self)
        data = np.array(self_attrs, dtype=header_dtype)
        data.tofile(file)


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

    @property
    def size(self):
        """The size in bytes of the header data."""
        self_attrs = attr.astuple(self, filter=EXCLUDE_NONE_FILTER)
        attrs_sizes = [self_attr.nbytes for self_attr in self_attrs]
        return sum(attrs_sizes)


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


@attr.s(auto_attribs=True)
class File(AbstractContextManager, MutableMapping):
    """Represent a GADGET-2 snapshot file."""

    path: os.PathLike
    mode: t.Optional[str] = "r"
    format: t.Optional[FileFormat] = None
    data_loaders: T_DataLoaders = attr.ib(default=None)
    _file: T_BinaryIO = attr.ib(default=None, init=False, repr=False)
    _header: Header = attr.ib(default=None, init=False, repr=False)
    _struct: T_Struct = attr.ib(default=None, init=False, repr=False)
    _temp_storage: T_TempStorage = attr.ib(default=None, init=False, repr=False)

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        mode = self.mode
        if mode not in FILE_MODES:
            raise ValueError(f"invalid mode; mode must be one of "
                             f"{FILE_MODES}")
        # Force the file to be open in binary mode.
        mode += "b"
        # noinspection PyTypeChecker
        file: T_BinaryIO = open(self.path, mode)
        object.__setattr__(self, "_file", file)
        # ************ Define the snapshot structure ************
        _format = self.format
        if not self.size:
            # Empty, writable files will have FileFormat.ALT.
            if _format is None:
                object.__setattr__(self, "format", FileFormat.ALT)
            object.__setattr__(self, "_struct", {})
            object.__setattr__(self, "_temp_storage", {})
        else:
            # ************ Define the snapshot Format ************
            if _format is not None:
                msg = f"can not set format '{_format}' to a nonempty snapshot"
                raise FormatError(msg)
            detected_format = self._detect_format()
            object.__setattr__(self, "format", detected_format)
            # ************ Initialize the header ************
            block_size = self._read_size_from_delim()
            header = Header.from_file(file)
            # Skip the remaining header bytes.
            self._skip(size=block_size - header.size)
            self._skip_block_delim()
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
            object.__setattr__(self, "_temp_storage", {})

    @property
    def name(self):
        return self._file.name

    @property
    def header(self):
        """The snapshot header."""
        return self._header

    @header.setter
    def header(self, new_header: Header):
        """Set the header in an empty snapshot."""
        if not self.is_empty():
            err_msg = "a nonempty snapshot header can not be changed."
            raise AttributeError(err_msg)
        self._header = new_header

    @property
    def size(self):
        """Size of the snapshot in bytes"""
        act_pos = self._file.tell()
        file_size = self._goto_end()
        self._file.seek(act_pos, io.SEEK_SET)
        return file_size

    def is_empty(self):
        """Test if this snapshot is empty."""
        return not self.size

    def flush(self):
        """Save in-memory block data to file."""
        if self.is_empty():
            if self.header is None:
                err_msg = "the snapshot header has not been initialized."
                raise AttributeError(err_msg)
            self._write_header(self.header)
        block_ids = list(self._temp_storage.keys())
        for block_id in block_ids:
            # Remove block from temporary storage.
            block = self._temp_storage.pop(block_id)
            block_data = block.data
            self._write_block_data(block_id, block_data)
            # Set the block id in the structure.
            self._struct[block_id] = None

    def close(self):
        """Close snapshot."""
        self._file.close()

    def _detect_format(self):
        """Detect the snapshot file format.

        :return: The snapshot format.
        """
        self__file = self._file
        size = self._read_size_from_delim()
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
            self._skip_block_delim()
            data_stream_pos = self__file.tell()
        else:
            # This is the data block. There is no additional block to read
            # the data block ID from.
            _id = None
            total_size = size + 2 * BLOCK_DELIM_SIZE
            # Return to the start of the data block.
            self._skip_block_delim(reverse=True)
            data_stream_pos = self__file.tell()
        return BlockSpec(total_size, data_stream_pos, _id)

    def _skip_block_delim(self, reverse: bool = False):
        """Skip a delimiter block.

        :param reverse: Skip the block backwards.
        """
        size = -BLOCK_DELIM_SIZE if reverse else BLOCK_DELIM_SIZE
        self._file.seek(size, io.SEEK_CUR)

    def _write_block_delim(self, block_size: int):
        """Write a delimiter block.

        :param block_size: The size in bytes of the delimited block.
        :return:
        """
        size_bytes = block_size.to_bytes(BLOCK_DELIM_SIZE, sys.byteorder)
        self._file.write(size_bytes)

    def _skip(self, size: int, reverse: bool = False):
        """Skip a block of ``size`` bytes.

        :param size: Size of block in bytes.
        :param reverse: Skip the block backwards.
        """
        size = -size if reverse else size
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
        # The structure must contain only block specs whose
        # ids belong to a well-defined data loader.
        return {block_id: spec_map[block_id] for block_id in loader_ids if
                block_id in spec_map}

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
        num_par_total = num_par_spec.total
        num_items = num_par_total * 3
        data: np.ndarray = np.fromfile(file, dtype="f4", count=num_items)
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
        num_par_spec = header.num_par_spec
        num_items = num_par_spec.total
        data: np.ndarray = np.fromfile(file, dtype="i4", count=num_items)
        num_par_spec_dict = attr.asdict(num_par_spec)
        return split_block_data(data, num_par_spec_dict)

    def _write_id_block(self, block_id: str, block_size: int):
        """Write a small identifier block.

        :param block_id: The id of the block being identified.
        :param block_size: The size of the block.
        :return:
        """
        self__file = self._file
        _ics = ID_CHUNK_SIZE
        total_size = block_size + 2 * BLOCK_DELIM_SIZE
        self._write_block_delim(ALT_ID_BLOCK_SIZE)
        size_bytes = total_size.to_bytes(SIZE_CHUNK_SIZE, sys.byteorder)
        id_bytes = f"{block_id:{_ics}.{_ics}}".encode("ascii")
        self__file.write(id_bytes + size_bytes)
        self._write_block_delim(ALT_ID_BLOCK_SIZE)

    def _write_header(self, header: Header):
        """Write the header data to an empty snapshot.

        :param header: The snapshot header.
        """
        if self.format is FileFormat.ALT:
            # Write identifier block.
            self._write_id_block("HEAD", HEADER_SIZE)
        self._write_block_delim(HEADER_SIZE)
        header.to_file(self._file)
        # Fill remaining header bytes with random data.
        random_bytes = os.urandom(HEADER_SIZE - header.size)
        self._file.write(random_bytes)
        self._write_block_delim(HEADER_SIZE)

    def _write_block_data(self, block_id: str, block_data: BlockData):
        """Write a block data to an empty snapshot.

        :param block_id: The id of the block whose data is being stored,
        :param block_data: The block data object.
        :return:
        """
        self__file = self._file
        eff_size = block_data.size
        if self.format is FileFormat.ALT:
            self._write_id_block(block_id, eff_size)
        self._write_block_delim(eff_size)
        data_attrs: t.Dict[str, np.ndarray] = \
            attr.asdict(block_data, filter=EXCLUDE_NONE_FILTER)
        for data_attr in data_attrs.values():
            data_attr.tofile(self__file)
        self._write_block_delim(eff_size)

    def keys(self):
        """The keys of the snapshot as a Mapping type instance."""
        return self._struct.keys()

    def __contains__(self, block_id: str):
        """Test if a block is present in this snapshot."""
        if block_id in self._temp_storage:
            return True
        return block_id in self._struct.keys()

    def __getitem__(self, block_id: str) -> Block:
        """Return the corresponding block."""
        if block_id not in self:
            raise KeyError(f"'{block_id}'")
        if block_id == "HEAD":
            msg = "header must be accessed through the 'header' attribute."
            raise ValueError(msg)
        if block_id in self._temp_storage:
            return self._temp_storage[block_id]
        data_loader = self.data_loaders[block_id]
        block_spec = self._struct[block_id]
        # NOTE: These checks seem unnecessary now...
        if data_loader is None:
            err_msg = f"loader for block '{block_id}' is not defined"
            raise TypeError(err_msg)
        if block_spec is None:
            err_msg = f"block '{block_id}' not found in snapshot"
            raise FormatError(err_msg)
        self._goto_block(block_spec)
        block_size = self._read_size_from_delim()
        block_data_dict = data_loader(self._file, self.header)
        block_data = BlockData(**block_data_dict)
        self._skip_block_delim()
        assert block_size == block_data.size
        return Block(block_id, block_data)

    def __setitem__(self, block_id: str, block: Block):
        """Save a block and add it to the snapshot structure."""
        if not isinstance(block_id, str):
            raise KeyError("the block id must be a string")
        if block_id == "HEAD":
            msg = "header must be set through the 'header' attribute"
            raise KeyError(msg)
        if block_id in self._struct:
            msg = f"a nonempty snapshot does not support blocks " \
                  f"reassignment (block '{block_id}' already set)"
            raise KeyError(msg)
        assert block.id == block_id
        self._temp_storage[block_id] = block

    def __delitem__(self, block_id: str):
        """Delete a block"""
        if block_id in self._temp_storage:
            del self._temp_storage[block_id]
            return
        msg = f"a nonempty snapshot does not support blocks deletion"
        raise KeyError(msg)

    def __len__(self) -> int:
        return len(self._struct) + len(self._temp_storage)

    def __iter__(self):
        struct_block_ids = self._struct.keys()
        temp_storage_block_ids = self._temp_storage.keys()
        return chain(struct_block_ids, temp_storage_block_ids)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
