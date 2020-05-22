import io
import os
import sys
import typing as t
from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
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
T_DataHandlers = t.Dict[str, "DataHandler"]
T_Struct = t.Dict[str, t.Union["BlockSpec", None]]
T_BlockDataBuffer = t.Dict[str, np.ndarray]
T_DataBuffer = t.Dict[str, T_BlockDataBuffer]

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
# TODO: Add the remaining fields according to GADGET-2 manual.
header_dtype = np.dtype([
    ("Npart", "i4", 6),
    ("Massarr", "f8", 6),
    ("Time", "f8"),
    ("Redshift", "f8"),
    ("FlagSfr", "i4"),
    ("FlagFeedback", "i4"),
    ("Nall", "u4", 6),
    ("FlagCooling", "i4"),
    ("NumFiles", "i4"),
    ("BoxSize", "f8"),
    ("Omega0", "f8"),
    ("OmegaLambda", "f8"),
    ("HubbleParam", "f8")
])


@attr.s(auto_attribs=True)
class ParSpec:
    """The spec of a single particle type."""
    num: int
    mass: float
    total_num: int


@attr.s(auto_attribs=True)
class ParSpecs:
    """Particles information."""
    gas: ParSpec
    halo: ParSpec
    disk: ParSpec
    bulge: ParSpec
    stars: ParSpec
    bndry: ParSpec


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
class ParMassSpec:
    gas: float
    halo: float
    disk: float
    bulge: float
    starts: float
    bndry: float


@attr.s(auto_attribs=True)
class Header:
    """Snapshot Header."""
    num_pars: NumParSpec
    par_masses: ParMassSpec
    time: float
    redshift: float
    flag_sfr: int
    flag_feedback: int
    total_num_pars: NumParSpec
    flag_cooling: int
    num_files_snap: int
    box_size: float
    omega_zero: float
    omega_lambda: float
    hubble_param: float

    @property
    def size(self):
        """Size of the header data in bytes."""
        return header_dtype.itemsize

    @property
    def par_specs(self):
        num_pars = attr.astuple(self.num_pars)
        par_masses = attr.astuple(self.par_masses)
        total_num_pars = attr.astuple(self.total_num_pars)
        par_spec_data = zip(num_pars, par_masses, total_num_pars)
        return ParSpecs(*(ParSpec(*par_spec) for par_spec in par_spec_data))

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
        num_pars = NumParSpec(*data['Npart'])
        par_masses = ParMassSpec(*data["Massarr"])
        total_num_pars = NumParSpec(*data["Nall"])
        header = cls(num_pars=num_pars,
                     par_masses=par_masses,
                     time=float(data["Time"]),
                     redshift=float(data["Redshift"]),
                     flag_sfr=bool(data["FlagSfr"]),
                     flag_feedback=bool(data["FlagFeedback"]),
                     total_num_pars=total_num_pars,
                     flag_cooling=bool(data["FlagCooling"]),
                     num_files_snap=int(data["NumFiles"]),
                     box_size=float(data["BoxSize"]),
                     omega_zero=float(data["Omega0"]),
                     omega_lambda=float(data["OmegaLambda"]),
                     hubble_param=float(data["HubbleParam"]))
        return header

    def to_file(self, file: T_BinaryIO):
        """Write the header contents to a file.

        :param file:
        :return:
        """
        self_attrs = attr.astuple(self)
        data = np.array(self_attrs, dtype=header_dtype)
        data.tofile(file)
        # Fill remaining header bytes with random data.
        random_bytes = os.urandom(HEADER_SIZE - self.size)
        file.write(random_bytes)


@attr.s(auto_attribs=True)
class BlockSpec:
    """"""
    total_size: int
    data_stream_pos: int
    id: str = None


@attr.s(auto_attribs=True, frozen=True, repr=False)
class BlockElement:
    """The block data of a specific particle type."""
    type: str
    block_id: str
    loader_func: t.Callable[[str, str], t.Optional[np.ndarray]]

    @property
    def data(self):
        """The data element as a numpy array instance."""
        return self.loader_func(self.block_id, self.type)

    def __repr__(self):
        cls_name = self.__class__.__name__
        par_type = self.type.capitalize()
        return f"<Snapshot {cls_name} instance; " \
               f"particle type: '{par_type}; block ID: {self.block_id}'>"


@attr.s(auto_attribs=True, frozen=True, repr=False)
class Block(Mapping):
    """A snapshot data block."""
    id: str
    par_types: t.Tuple[str, ...]
    loader_func: t.Callable[[str, str], np.ndarray]

    def __contains__(self, key):
        return key in self.par_types

    def __getitem__(self, key: str) -> BlockElement:
        if key not in self:
            raise KeyError(f"'{key}'")
        return BlockElement(key, self.id, self.loader_func)

    def __iter__(self) -> t.Iterator[str]:
        return iter(self.par_types)

    def __len__(self) -> int:
        return len(self.par_types)

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"<Snapshot {cls_name} instance; block ID: '{self.id}'>"


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


@attr.s(auto_attribs=True, frozen=True, repr=False)
class DataHandler:
    """Base class for data loaders."""

    # Type annotations. They should be defined as attributes or \
    # properties in derived classes.
    par_specs: t.Dict[str, ParSpec]
    dtype: t.Union[str, np.dtype]
    par_items: int

    def __attrs_post_init__(self):
        """Post-init stage. Intended for attrs classes."""
        dtype = self.dtype
        if isinstance(dtype, str):
            object.__setattr__(self, "dtype", np.dtype(dtype))

    @property
    def item_size(self):
        """The size in bytes of a single data item."""
        return self.dtype.itemsize

    @property
    def sizes(self) -> t.Dict[str, int]:
        """Size (in bytes) of the data used to define a particle state."""
        par_size = self.item_size * self.par_items
        par_types, par_specs = zip(*self.par_specs.items())
        sizes = (par_spec.num * par_size for par_spec in par_specs)
        return dict(zip(par_types, sizes))

    @property
    def offsets(self) -> t.Dict[str, int]:
        """Size (in bytes) of the data used to define a particle state."""
        par_types, sizes = zip(*self.sizes.items())
        offsets = (0,) + tuple(accumulate(sizes[:-1]))
        return dict(zip(par_types, offsets))

    def __repr__(self):
        dtype = self.dtype
        cls_name = self.__class__.__name__
        par_items = self.par_items
        return f"<{cls_name}, dtype='{dtype}', items per particle: {par_items}>"

    @abstractmethod
    def read(self, file: T_BinaryIO, par_type: str,
             par_spec: ParSpec) -> np.ndarray:
        pass

    def write(self, file: T_BinaryIO, data_elem: np.ndarray):
        """Write particle data to a binary file.

        :param file: A binary file.
        :param data_elem: The particle data.
        :return: The number of bytes written.
        """
        data_array = data_elem.data
        if data_array is None:
            return 0
        data_array = np.asarray(data_array, dtype=self.dtype)
        data_array.tofile(file)
        return data_array.nbytes


@attr.s(auto_attribs=True, frozen=True, repr=False)
class PositionsLikeDataHandler(DataHandler):
    """Loader for reading positions-like data from a snapshot."""
    par_specs: t.Dict[str, ParSpec]
    dtype: np.dtype = "f4"
    par_items: int = 3

    def read(self, file: T_BinaryIO, par_type: str, par_spec: ParSpec):
        """Read data with a similar structure as positions.

        :param file: Snapshot file.
        :param par_type: The particle type.
        :param par_spec: The particle spec.
        :return: A numpy array with the data.
        """
        num_par = par_spec.num
        num_items = num_par * self.par_items
        data: np.ndarray = np.fromfile(file, dtype=self.dtype, count=num_items)
        return data.reshape((num_par, self.par_items))


@attr.s(auto_attribs=True, frozen=True, repr=False)
class IDsDataHandler(DataHandler):
    """Loader for reading IDs data from a snapshot."""
    par_specs: t.Dict[str, ParSpec]
    dtype: np.dtype = "i4"
    par_items: int = 1

    def read(self, file: T_BinaryIO, par_type: str, par_spec: ParSpec):
        """Read identifiers data from a binary file.

        :param file: Snapshot file.
        :param par_type: The particle type.
        :param par_spec: The particle spec.
        :return: A numpy array with the particle identifiers.
        """
        num_par = par_spec.num
        return np.fromfile(file, dtype=self.dtype, count=num_par)


@attr.s(auto_attribs=True)
class DataHandlers:
    """Classes used to load data from the snapshot blocks.

    This class defines the python callable objects used to load data from
    the blocks in a GADGET snapshot, according to the standard snapshot
    specification (see GADGET-2 manual, p.32).
    """
    POS: DataHandler = None
    VEL: DataHandler = None
    ID: DataHandler = None
    MASS: DataHandler = None
    U: DataHandler = None
    RHO: DataHandler = None
    HSML: DataHandler = None
    POT: DataHandler = None
    ACCE: DataHandler = None
    ENDT: DataHandler = None
    TSTP: DataHandler = None


@attr.s(auto_attribs=True)
class File(AbstractContextManager, MutableMapping):
    """Represent a GADGET-2 snapshot file."""

    path: os.PathLike
    mode: t.Optional[str] = "r"
    format: t.Optional[FileFormat] = None
    data_handlers: T_DataHandlers = \
        attr.ib(default=None, init=False, repr=False)
    stream: T_BinaryIO = attr.ib(default=None, init=False, repr=False)
    _header: Header = attr.ib(default=None, init=False, repr=False)
    _struct: T_Struct = attr.ib(default=None, init=False, repr=False)
    _data_buffer: T_DataBuffer = attr.ib(default=None, init=False, repr=False)

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        mode = self.mode
        if mode not in FILE_MODES:
            raise ValueError(f"invalid mode; mode must be one of "
                             f"{FILE_MODES}")
        # Force the file to be open in binary mode.
        mode += "b"
        # noinspection PyTypeChecker
        stream: T_BinaryIO = open(self.path, mode)
        object.__setattr__(self, "stream", stream)
        # ************ Define the snapshot structure ************
        _format = self.format
        if not self.size:
            # Empty, writable files will have FileFormat.ALT.
            if _format is None:
                object.__setattr__(self, "format", FileFormat.ALT)
            object.__setattr__(self, "_struct", {})
            object.__setattr__(self, "_data_buffer", {})
        else:
            # ************ Define the snapshot Format ************
            if _format is not None:
                msg = f"can not set format '{_format}' to a nonempty snapshot"
                raise FormatError(msg)
            detected_format = self._detect_format()
            object.__setattr__(self, "format", detected_format)
            # ************ Initialize the header ************
            block_size = self._read_size_from_delim()
            header = Header.from_file(stream)
            # Skip the remaining header bytes.
            self._skip(size=block_size - header.size)
            self._skip_block_delim()
            stream.seek(SNAP_START_POS)
            object.__setattr__(self, "_header", header)
            # ************ Define the block data loaders **************
            # These are the default block data loaders
            data_handlers = self._init_data_handlers()
            object.__setattr__(self, "data_handlers", data_handlers)
            # ************ Load block specs ************
            struct: T_Struct = self._define_struct()
            object.__setattr__(self, "_struct", struct)
            object.__setattr__(self, "_data_buffer", {})

    @property
    def name(self):
        return self.stream.name

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
        self.data_handlers = self._init_data_handlers()

    @property
    def size(self):
        """Size of the snapshot in bytes"""
        cur_pos = self.stream.tell()
        file_size = self._goto_end()
        self.stream.seek(cur_pos, io.SEEK_SET)
        return file_size

    def is_empty(self):
        """Test if this snapshot is empty."""
        return not self.size

    def flush(self):
        """Save in-memory data buffers to file."""
        if self.header is None:
            err_msg = "the snapshot header has not been initialized."
            raise AttributeError(err_msg)
        if self.is_empty():
            # Only write the header if this file is empty.
            self._write_header(self.header)
        block_ids = list(self._data_buffer.keys())
        for block_id in block_ids:
            # Remove block from temporary storage.
            data_buffer = self._data_buffer.pop(block_id)
            self._write_block_data(block_id, data_buffer)
            # Set the block id in the structure.
            self._struct[block_id] = None

    def close(self):
        """Close snapshot."""
        self.flush()
        self.stream.close()

    def _init_data_handlers(self) -> T_DataHandlers:
        """"""
        par_specs = attr.asdict(self.header.par_specs, recurse=False)
        data_handlers = DataHandlers(
                POS=PositionsLikeDataHandler(par_specs),
                VEL=PositionsLikeDataHandler(par_specs),
                ID=IDsDataHandler(par_specs)
        )
        return attr.asdict(data_handlers, filter=EXCLUDE_NONE_FILTER,
                           recurse=False)

    def _detect_format(self):
        """Detect the snapshot file format.

        :return: The snapshot format.
        """
        stream = self.stream
        size = self._read_size_from_delim()
        if size not in [HEADER_SIZE, ALT_ID_BLOCK_SIZE]:
            # The first block can only have two possible sizes.
            raise FormatError("this is not a valid snapshot file")
        body_bytes = stream.read(size)
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
        skip_block_delim(stream)
        if _format is FileFormat.DEFAULT:
            # Reset stream position to the start.
            stream.seek(SNAP_START_POS, io.SEEK_SET)
        return _format

    def _goto_start(self):
        """Go to the snapshot starting position."""
        return self.stream.seek(SNAP_START_POS, io.SEEK_SET)

    def _goto_end(self):
        """Go to the snapshot ending position."""
        return self.stream.seek(SNAP_START_POS, io.SEEK_END)

    def _read_size_from_delim(self):
        """Read a delimiter block and return its contents. The returned value
        is the size in bytes of the following data block.

        :return: Size of following data block in bytes.
        """
        size_bytes = self.stream.read(BLOCK_DELIM_SIZE)
        if size_bytes == b"":
            raise SnapshotEOFError
        return int.from_bytes(size_bytes, sys.byteorder)

    def _read_block_spec(self):
        """Load the identified and size of a snapshot block.

        :return:
        """
        stream = self.stream
        size = self._read_size_from_delim()
        if self.format is FileFormat.ALT:
            # Read the block ID from the additional block
            body_bytes = stream.read(size)
            id_bytes = body_bytes[:ID_CHUNK_SIZE].decode("ascii")
            _id = str(id_bytes).rstrip()
            # Get the total size (including delimiter blocks) of
            # the block's data.
            total_size_bytes = body_bytes[ID_CHUNK_SIZE:]
            total_size = int.from_bytes(total_size_bytes, sys.byteorder)
            self._skip_block_delim()
            data_stream_pos = stream.tell()
        else:
            # This is the data block. There is no additional block to read
            # the data block ID from.
            _id = None
            total_size = size + 2 * BLOCK_DELIM_SIZE
            # Return to the start of the data block.
            self._skip_block_delim(reverse=True)
            data_stream_pos = stream.tell()
        return BlockSpec(total_size, data_stream_pos, _id)

    def _skip_block_delim(self, reverse: bool = False):
        """Skip a delimiter block.

        :param reverse: Skip the block backwards.
        """
        size = -BLOCK_DELIM_SIZE if reverse else BLOCK_DELIM_SIZE
        self.stream.seek(size, io.SEEK_CUR)

    def _write_block_delim(self, block_size: int):
        """Write a delimiter block.

        :param block_size: The size in bytes of the delimited block.
        :return:
        """
        size_bytes = block_size.to_bytes(BLOCK_DELIM_SIZE, sys.byteorder)
        self.stream.write(size_bytes)

    def _skip(self, size: int, reverse: bool = False):
        """Skip a block of ``size`` bytes.

        :param size: Size of block in bytes.
        :param reverse: Skip the block backwards.
        """
        size = -size if reverse else size
        self.stream.seek(size, io.SEEK_CUR)

    def _skip_block(self, block_spec: BlockSpec):
        """Skip a block of ``block_spec` bytes.

        :param self: Snapshot file.
        :param block_spec: The block spec.
        """
        total_size = block_spec.total_size
        data_stream_pos = block_spec.data_stream_pos
        self.stream.seek(data_stream_pos + total_size, io.SEEK_SET)

    def inspect(self):
        """Inspect the structure of a snapshot."""
        return list(self._inspect_struct())

    def _inspect_struct(self):
        """Inspect the structure of a snapshot."""
        # Read snapshot header spec.
        try:
            self.stream.seek(SNAP_START_POS, io.SEEK_SET)
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
        handler_ids = list(self.data_handlers.keys())

        def patch_spec(spec: BlockSpec, type_id: str):
            """Set the ID string of a BlockSpec manually."""
            return attr.evolve(spec, id=type_id)

        if self.format is FileFormat.DEFAULT:
            specs_and_loader_ids = zip(spec_list, handler_ids)
            spec_list = starmap(patch_spec, specs_and_loader_ids)
            spec_map: T_Struct = dict(zip(handler_ids, spec_list))
        else:
            spec_map = {spec.id: spec for spec in spec_list}
        # The structure must contain only block specs whose
        # ids belong to a well-defined data loader.
        return {block_id: spec_map[block_id] for block_id in handler_ids if
                block_id in spec_map}

    def _goto_block(self, block_spec: BlockSpec):
        """Jump directly to the block data location in the snapshot.

        :param block_spec: The block spec.
        """
        data_stream_pos = block_spec.data_stream_pos
        self.stream.seek(data_stream_pos, io.SEEK_SET)

    def create_block(self, block_id: str, data: t.Mapping[str, t.Any]):
        """Create a new block in the snapshot.

        :param block_id: The block ID.
        :param data: The block data object.
        :return: A new Block instance.
        """
        par_specs = attr.asdict(self.header.par_specs)
        # Create a dict with a key for each particle type but no data.
        data_buffer = {par_type: None for par_type in par_specs}
        for par_type in data_buffer.keys():
            # Update the block data.
            data_buffer[par_type] = data.get(par_type, None)
        self._data_buffer[block_id] = data_buffer
        return self[block_id]

    def load_block_par_data(self, block_id: str, par_type: str):
        """Load positions data from a binary stream.

        :param block_id: The block ID.
        :param par_type: The particle type.
        :return: The particle data as a numpy array.
        """
        stream = self.stream
        header = self.header
        data_loader = self.data_handlers[block_id]
        block_spec = self._struct[block_id]
        if block_id not in self:
            raise KeyError(f"'{block_id}'")
        # NOTE: These checks seem unnecessary now...
        if data_loader is None:
            err_msg = f"loader for block '{block_id}' is not defined"
            raise TypeError(err_msg)
        if block_spec is None:
            err_msg = f"block '{block_id}' not found in snapshot"
            raise FormatError(err_msg)
        if block_id in self._data_buffer:
            return self._data_buffer[block_id][par_type]
        par_specs_dict = attr.asdict(header.par_specs, recurse=False)
        par_spec: ParSpec = par_specs_dict[par_type]
        offset = data_loader.offsets[par_type]
        # Jump to the current block position.
        self._goto_block(block_spec)
        self._skip_block_delim()
        # Jump to the data location starting from the current position.
        stream.seek(offset, io.SEEK_CUR)
        if not par_spec.num:
            return None
        data = data_loader.read(stream, par_type, par_spec)
        assert data.nbytes == data_loader.sizes[par_type]
        return data

    def _write_id_block(self, block_id: str, block_size: int):
        """Write a small identifier block.

        :param block_id: The id of the block being identified.
        :param block_size: The size of the block.
        :return:
        """
        stream = self.stream
        _ics = ID_CHUNK_SIZE
        total_size = block_size + 2 * BLOCK_DELIM_SIZE
        self._write_block_delim(ALT_ID_BLOCK_SIZE)
        size_bytes = total_size.to_bytes(SIZE_CHUNK_SIZE, sys.byteorder)
        id_bytes = f"{block_id:{_ics}.{_ics}}".encode("ascii")
        stream.write(id_bytes + size_bytes)
        self._write_block_delim(ALT_ID_BLOCK_SIZE)

    def _write_header(self, header: Header):
        """Write the header data to an empty snapshot.

        :param header: The snapshot header.
        """
        if self.format is FileFormat.ALT:
            # Write identifier block.
            self._write_id_block("HEAD", HEADER_SIZE)
        self._write_block_delim(HEADER_SIZE)
        header.to_file(self.stream)
        self._write_block_delim(HEADER_SIZE)

    def _write_block_data(self, block_id: str,
                          data_buffer: T_BlockDataBuffer):
        """Write a block data to an empty snapshot.

        :param block_id: The id of the block whose data is being stored,
        :param data_buffer: The block data object.
        :return:
        """
        stream = self.stream
        data_handler = self.data_handlers[block_id]
        ini_pos = stream.tell()
        # We can't know the block size a priori. We write a temporary
        # id block, which we will update later.
        if self.format is FileFormat.ALT:
            self._write_id_block(block_id, 0)
        self._write_block_delim(0)
        block_size = 0
        for par_type in data_buffer:
            data_elem = data_buffer[par_type]
            if data_elem is None:
                continue
            block_size += data_handler.write(stream, data_elem)
        final_pos = stream.tell()
        # We have to overwrite the ID block and the block delimiter to
        # set the correct block size. It is not elegant, but currently,
        # I can't think about a better approach.
        stream.seek(ini_pos, io.SEEK_SET)
        if self.format is FileFormat.ALT:
            self._write_id_block(block_id, block_size)
        self._write_block_delim(block_size)
        # Jump to the final position and close the block.
        stream.seek(final_pos, io.SEEK_SET)
        self._write_block_delim(block_size)

    def keys(self):
        """The keys of the snapshot as a Mapping type instance."""
        return self._struct.keys()

    def __contains__(self, block_id: str):
        """Test if a block is present in this snapshot."""
        if block_id in self._data_buffer:
            return True
        return block_id in self._struct.keys()

    def __getitem__(self, block_id: str) -> Block:
        """Return the corresponding block."""
        if block_id not in self:
            raise KeyError(f"'{block_id}'")
        if block_id == "HEAD":
            msg = "header must be accessed through the 'header' attribute."
            raise ValueError(msg)
        par_types = tuple(attr.asdict(self.header.par_specs, recurse=False))
        return Block(block_id, par_types, self.load_block_par_data)

    def __setitem__(self, block_id: str, data: t.Any):
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
        # TODO: Should we avoid the insertion of blocks that do not
        #   have a well-defined data loader?
        self.create_block(block_id, data)

    def __delitem__(self, block_id: str):
        """Delete a block"""
        if block_id in self._data_buffer:
            del self._data_buffer[block_id]
            return
        msg = f"a nonempty snapshot does not support blocks deletion"
        raise KeyError(msg)

    def __len__(self) -> int:
        return len(self._struct) + len(self._data_buffer)

    def __iter__(self):
        struct_block_ids = self._struct.keys()
        temp_storage_block_ids = self._data_buffer.keys()
        return chain(struct_block_ids, temp_storage_block_ids)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
