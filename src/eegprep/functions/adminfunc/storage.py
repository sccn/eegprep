"""Python-native storage helpers for EEGLAB-style large datasets."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS

FDT_DTYPE = np.dtype("<f4")


class _MutationTrackedFlat:
    """Flat iterator wrapper that notifies after indexed writes."""

    def __init__(self, array: np.ndarray, on_mutation: Callable[[], None]) -> None:
        self._flat = array.flat
        self._on_mutation = on_mutation

    def __getitem__(self, key: Any) -> Any:
        return self._flat[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._flat[key] = value
        self._on_mutation()

    def __iter__(self) -> Any:
        return iter(self._flat)

    def __len__(self) -> int:
        return len(self._flat)


class _MutationTrackedArray(np.ndarray):
    """Array view that invokes a callback after an in-place write."""

    _on_mutation: Callable[[], None] | None

    def __array_finalize__(self, source: Any) -> None:
        self._on_mutation = getattr(source, "_on_mutation", None)

    def _notify(self) -> None:
        if self._on_mutation is not None:
            self._on_mutation()

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self._notify()

    def fill(self, value: Any) -> None:
        super().fill(value)
        self._notify()

    @property
    def flat(self) -> _MutationTrackedFlat:
        """Return a one-dimensional view that preserves write tracking."""
        return _MutationTrackedFlat(self, self._notify)

    def sort(self, *args: Any, **kwargs: Any) -> None:
        super().sort(*args, **kwargs)
        self._notify()

    def partition(self, *args: Any, **kwargs: Any) -> None:
        super().partition(*args, **kwargs)
        self._notify()

    def byteswap(self, inplace: bool = False) -> np.ndarray:
        result = super().byteswap(inplace=inplace)
        if inplace:
            self._notify()
        return result

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        outputs = kwargs.get("out")
        mutating_input = inputs[0] if method == "at" and inputs else None
        tracked_outputs = outputs and any(isinstance(output, _MutationTrackedArray) for output in outputs)
        if outputs:
            kwargs["out"] = tuple(
                np.asarray(output) if isinstance(output, _MutationTrackedArray) else output for output in outputs
            )
        inputs = tuple(np.asarray(value) if isinstance(value, _MutationTrackedArray) else value for value in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        if method == "at" and isinstance(mutating_input, _MutationTrackedArray):
            mutating_input._notify()
        elif method == "__call__" and tracked_outputs:
            for output in outputs:
                if isinstance(output, _MutationTrackedArray):
                    output._notify()
        return result

    def __array_function__(self, function: Any, types: Any, args: Any, kwargs: Any) -> Any:  # ty: ignore[invalid-method-override]
        if function is np.copyto and args and isinstance(args[0], _MutationTrackedArray):
            converted_args = tuple(
                np.asarray(value) if isinstance(value, _MutationTrackedArray) else value for value in args
            )
            result = np.copyto(*converted_args, **kwargs)
            args[0]._notify()
            return result
        if function is np.put and args and isinstance(args[0], _MutationTrackedArray):
            converted_args = tuple(
                np.asarray(value) if isinstance(value, _MutationTrackedArray) else value for value in args
            )
            result = np.put(*converted_args, **kwargs)
            args[0]._notify()
            return result
        return super().__array_function__(function, types, args, kwargs)


class _BackingFile:
    """Reference-counted backing path shared by copy-on-write handles."""

    def __init__(self, path: Path, *, temporary: bool = False) -> None:
        self.path = path
        self.temporary = temporary
        self.references = 0

    def acquire(self) -> None:
        self.references += 1

    def release(self) -> None:
        self.references -= 1
        if self.references == 0 and self.temporary:
            self.path.unlink(missing_ok=True)


def _unwrap_memmap_data(value: Any) -> Any:
    """Replace MemmapData handles with their mapped arrays, including inside containers.

    Nested handles have to be unwrapped too. Leaving one inside a list would make numpy
    dispatch straight back into ``MemmapData.__array_function__`` and recurse forever on
    calls like ``np.concatenate([mapped_a, mapped_b])``.
    """
    if isinstance(value, MemmapData):
        return value._memmap()
    if isinstance(value, dict):
        return {key: _unwrap_memmap_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap_memmap_data(item) for item in value)
    return value


class MemmapData:
    """NumPy-compatible handle for channel-major EEG data stored on disk.

    Indexing always uses the logical EEG shape and normal zero-based NumPy
    indices. ``transposed=True`` maps EEGLAB ``.dat``-style storage whose
    physical axes are ``(samples, trials, channels)`` without exposing that
    layout to callers. Explicit and deep copies share the backing file until a
    copy is mutated, at which point that handle gets a private temporary file.
    """

    __array_priority__ = 1000

    def __init__(
        self,
        filename: str | Path,
        shape: tuple[int, ...],
        *,
        dtype: np.dtype | str = FDT_DTYPE,
        mode: str = "r+",
        order: str = "F",
        transposed: bool = False,
        temporary: bool = False,
        debug: bool = False,
        _backing: _BackingFile | None = None,
    ) -> None:
        self._shape = tuple(int(item) for item in shape)
        if not self._shape or any(item <= 0 for item in self._shape):
            raise ValueError("Memory-mapped data dimensions must be positive")
        self._dtype = np.dtype(dtype)
        if mode not in {"r", "r+", "c"}:
            raise ValueError("MemmapData mode must be 'r', 'r+', or 'c'")
        if order not in {"C", "F"}:
            raise ValueError("MemmapData order must be 'C' or 'F'")
        self.mode = mode
        self.order = order
        self.transposed = bool(transposed)
        if self.transposed and len(self._shape) not in {2, 3}:
            raise ValueError("Transposed storage requires a two- or three-dimensional shape")
        self.debug = bool(debug)
        self.type = "mmo"
        self._backing = _backing or _BackingFile(Path(filename), temporary=temporary)
        self._backing.acquire()
        self._released = False
        self._array: np.memmap | None = None
        self._mutation_revision = 0
        self._validate_backing_file()

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        *,
        filename: str | Path | None = None,
        dtype: np.dtype | str = FDT_DTYPE,
        order: str = "F",
        transposed: bool = False,
        temporary: bool | None = None,
        fill_value: float = 0.0,
    ) -> "MemmapData":
        """Create a writable disk-backed array initialized to ``fill_value``."""
        normalized_shape = tuple(int(item) for item in shape)
        if not normalized_shape or any(item <= 0 for item in normalized_shape):
            raise ValueError("Memory-mapped data dimensions must be positive")
        if transposed and len(normalized_shape) not in {2, 3}:
            raise ValueError("Transposed storage requires a two- or three-dimensional shape")
        path, owns_file = _output_mapping_path(filename)
        if temporary is None:
            temporary = owns_file
        physical_shape = _physical_shape(normalized_shape, transposed)
        mapped = np.memmap(path, dtype=np.dtype(dtype), mode="w+", shape=physical_shape, order=order)
        mapped[...] = fill_value
        mapped.flush()
        mmap_handle = getattr(mapped, "_mmap", None)
        del mapped
        if mmap_handle is not None:
            mmap_handle.close()
        return cls(
            path,
            normalized_shape,
            dtype=dtype,
            mode="r+",
            order=order,
            transposed=transposed,
            temporary=bool(temporary),
        )

    @classmethod
    def from_array(
        cls,
        data: Any,
        *,
        filename: str | Path | None = None,
        dtype: np.dtype | str = FDT_DTYPE,
        order: str = "F",
        transposed: bool = False,
        temporary: bool | None = None,
    ) -> "MemmapData":
        """Write an array to disk and return a logical memory-mapped view."""
        array = np.asarray(data, dtype=np.dtype(dtype))
        if array.ndim == 0 or any(item <= 0 for item in array.shape):
            raise ValueError("Memory-mapped data must have positive dimensions")
        output = cls.empty(
            tuple(array.shape),
            filename=filename,
            dtype=dtype,
            order=order,
            transposed=transposed,
            temporary=temporary,
        )
        output._memmap()[...] = array
        output.flush()
        return output

    @property
    def filename(self) -> str:
        """Return the backing file path as a string."""
        return str(self.path)

    @property
    def path(self) -> Path:
        """Return the backing file path."""
        return self._backing.path

    @property
    def dataFile(self) -> str:
        """Return the backing path using EEGLAB's ``mmo`` field name."""
        return self.filename

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Return the logical dimensions using EEGLAB's ``mmo`` field name."""
        return self.shape

    @property
    def writable(self) -> bool:
        """Return whether writes are permitted for this handle."""
        return self.mode != "r"

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the EEG-shaped array dimensions."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Return the backing file dtype."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Return the number of EEG data dimensions."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Return the total sample count."""
        return int(np.prod(self._shape))

    @property
    def mutation_revision(self) -> int:
        """Return the in-process write revision for this mapped dataset."""
        return self._mutation_revision

    @property
    def T(self) -> np.ndarray:
        """Return a transposed array view."""
        return self._tracked_view(self._memmap().T)

    @property
    def flat(self) -> _MutationTrackedFlat:
        """Return a one-dimensional view that preserves write tracking."""
        return _MutationTrackedFlat(self._memmap(), self._mark_mutated)

    def fill(self, value: Any) -> None:
        """Fill the mapped data and advance its mutation revision."""
        self._memmap().fill(value)
        self._mark_mutated()

    def flush(self) -> None:
        """Flush pending writes to the backing file."""
        self._memmap()
        if self._array is not None:
            self._array.flush()

    def close(self) -> None:
        """Flush and release the backing memory map handle."""
        array = self._array
        if array is None:
            return
        self._array = None
        array.flush()
        # Do not close ``array._mmap`` directly: NumPy slices can retain views
        # of this map after the handle object is replaced inside an EEG dict.
        # Releasing our reference lets NumPy close the OS map once every view
        # is gone and avoids invalidating those still-live arrays.

    def copy(self, order: str = "C") -> np.ndarray:
        """Return an in-memory copy of the mapped data."""
        return np.array(self._memmap(), copy=True, order=order)

    def mapped_copy(self) -> "MemmapData":
        """Return an independent writable disk-backed copy."""
        copied = self.__copy__()
        copied._detach_for_write()
        return copied

    def resize(self, shape: tuple[int, ...], *, fill_value: float = 0.0) -> None:
        """Resize this mapping, retaining the overlapping logical data region.

        Resizing writes a private replacement sidecar. The original sidecar is
        never resized underneath another handle.
        """
        new_shape = tuple(int(item) for item in shape)
        if not new_shape or any(item <= 0 for item in new_shape):
            raise ValueError("Memory-mapped data dimensions must be positive")
        if len(new_shape) != self.ndim:
            raise ValueError("Resizing cannot change the number of dimensions")
        if self.transposed and len(new_shape) not in {2, 3}:
            raise ValueError("Transposed storage requires a two- or three-dimensional shape")
        replacement = MemmapData.empty(
            new_shape,
            dtype=self.dtype,
            order=self.order,
            transposed=self.transposed,
            fill_value=fill_value,
        )
        common_region = tuple(slice(0, min(self.shape[axis], new_shape[axis])) for axis in range(self.ndim))
        replacement[common_region] = self[common_region]
        self._adopt(replacement)

    def delete(self, indices: Any, *, axis: int | None = None) -> None:
        """Delete logical indices and replace the sidecar with the smaller data.

        ``axis=None`` follows MATLAB's column-major linear deletion. Matrix
        results become row vectors, while column-vector inputs remain columns.
        """
        if axis is None:
            flat = np.asarray(self).reshape(-1, order="F")
            result = np.delete(flat, indices)
            if self.ndim == 2 and self.shape[1] == 1:
                result = result.reshape((-1, 1), order="F")
            else:
                result = result.reshape((1, -1), order="F")
        else:
            normalized_axis = int(axis)
            if normalized_axis < -self.ndim or normalized_axis >= self.ndim:
                raise ValueError(f"axis {normalized_axis} is out of bounds for array of dimension {self.ndim}")
            result = np.delete(np.asarray(self), indices, axis=normalized_axis)
        if result.size == 0:
            raise ValueError("MemmapData cannot represent an empty mapping")
        replacement = MemmapData.from_array(
            result,
            dtype=self.dtype,
            order=self.order,
            transposed=self.transposed and result.ndim in {2, 3},
        )
        self._adopt(replacement)

    def change_file(self, filename: str | Path, *, writable: bool = False) -> None:
        """Move a private mapping, or copy a shared mapping, to ``filename``."""
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.flush()
        source = self.path
        shared = self._backing.references > 1
        self.close()
        if shared:
            shutil.copyfile(source, target)
        else:
            shutil.move(source, target)
        self._replace_backing(_BackingFile(target), mode="r+" if writable else "r")

    def reshape(self, *shape: Any, **kwargs: Any) -> np.ndarray:
        """Return a reshaped view using NumPy's reshape semantics."""
        return self._tracked_view(self._memmap().reshape(*shape, **kwargs))

    def transpose(self, *axes: Any) -> np.ndarray:
        """Return a transposed view that preserves mutation tracking."""
        return self._tracked_view(self._memmap().transpose(*axes))

    def ravel(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return a flattened view or copy using NumPy's ravel semantics."""
        return self._tracked_view(self._memmap().ravel(*args, **kwargs))

    def sort(self, *args: Any, **kwargs: Any) -> None:
        """Sort mapped data in place and advance its mutation revision."""
        self._memmap().sort(*args, **kwargs)
        self._mark_mutated()

    def partition(self, *args: Any, **kwargs: Any) -> None:
        """Partition mapped data in place and advance its mutation revision."""
        self._memmap().partition(*args, **kwargs)
        self._mark_mutated()

    def byteswap(self, inplace: bool = False) -> np.ndarray:
        """Byte-swap mapped data while preserving mutation tracking."""
        result = self._memmap().byteswap(inplace=inplace)
        if inplace:
            self._mark_mutated()
            return self._tracked_view(result)
        return result

    def astype(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return a typed array using NumPy's astype semantics."""
        return self._memmap().astype(*args, **kwargs)

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        """Return NumPy's sum over the mapped data."""
        return self._memmap().sum(*args, **kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        """Return NumPy's mean over the mapped data."""
        return self._memmap().mean(*args, **kwargs)

    def __array__(self, dtype: np.dtype | str | None = None, copy: bool | None = None) -> np.ndarray:
        array = self._memmap()
        if dtype is not None:
            if np.dtype(dtype) != self._dtype or copy:
                return np.asarray(array, dtype=dtype).copy()
            return self._tracked_view(array)
        if copy:
            return np.asarray(array).copy()
        return self._tracked_view(array)

    def __getitem__(self, key: Any) -> Any:
        value = self._memmap()[key]
        return self._tracked_view(value) if isinstance(value, np.ndarray) else value

    def __setitem__(self, key: Any, value: Any) -> None:
        if not self.writable:
            raise ValueError("assignment destination is read-only")
        self._detach_for_write()
        self._memmap()[key] = value
        self._mark_mutated()

    def __array_function__(self, function: Any, types: Any, args: Any, kwargs: Any) -> Any:
        if function in {np.copyto, np.put} and args and args[0] is self:
            converted_args = (self._memmap(), *args[1:])
            result = function(*converted_args, **kwargs)
            self._mark_mutated()
            return result
        # Defining this method opts the type into numpy's dispatch protocol, which means
        # NotImplemented here does not fall back, it raises: every numpy function other than
        # the two above would fail on a MemmapData. That defeats a handle whose whole purpose
        # is to stand in for an array, and np.size(data) in pop_reref is one caller of many.
        # Delegate to the mapped array instead. The two mutating calls this class tracks are
        # handled above, and writes through a returned view are tracked by _tracked_view.
        return function(*_unwrap_memmap_data(args), **_unwrap_memmap_data(kwargs))

    def __len__(self) -> int:
        return len(self._memmap())

    def __iter__(self) -> Any:
        for index in range(len(self)):
            yield self[index]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._memmap(), name)

    def __copy__(self) -> "MemmapData":
        return MemmapData(
            self.path,
            self._shape,
            dtype=self._dtype,
            mode=self.mode,
            order=self.order,
            transposed=self.transposed,
            debug=self.debug,
            _backing=self._backing,
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "MemmapData":
        copied = self.__copy__()
        memo[id(self)] = copied
        return copied

    def __repr__(self) -> str:
        return (
            f"MemmapData(path={str(self.path)!r}, shape={self._shape!r}, "
            f"dtype={self._dtype}, transposed={self.transposed})"
        )

    def __del__(self) -> None:
        if getattr(self, "_released", True):
            return
        try:
            self.close()
        finally:
            self._backing.release()
            self._released = True

    def _memmap(self) -> np.ndarray:
        if self._array is None:
            self._array = np.memmap(
                self.path,
                dtype=self._dtype,
                mode=self.mode,
                shape=_physical_shape(self._shape, self.transposed),
                order=self.order,
            )
        if not self.transposed:
            return self._array
        if self.ndim == 2:
            return self._array.transpose(1, 0)
        return self._array.transpose(2, 0, 1)

    def _validate_backing_file(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Memory-mapped data file not found: {self.path}")
        expected = self.size * self.dtype.itemsize
        actual = self.path.stat().st_size
        if actual == 0:
            raise ValueError(f"Memory-mapped data file is empty: {self.path}")
        if actual != expected:
            raise ValueError(f"Memory-mapped data file has {actual} bytes, expected {expected}")

    def _detach_for_write(self) -> None:
        if self._backing.references <= 1:
            return
        self.flush()
        new_path, _owns_file = _output_mapping_path(None)
        shutil.copyfile(self.path, new_path)
        self.close()
        self._replace_backing(_BackingFile(new_path, temporary=True), mode="r+")

    def _adopt(self, replacement: "MemmapData") -> None:
        replacement.flush()
        self.close()
        self._shape = replacement.shape
        self._dtype = replacement.dtype
        self.order = replacement.order
        self.transposed = replacement.transposed
        self._replace_backing(replacement._backing, mode="r+")

    def _replace_backing(self, backing: _BackingFile, *, mode: str) -> None:
        old_backing = self._backing
        backing.acquire()
        self._backing = backing
        self.mode = mode
        self._array = None
        old_backing.release()

    def _mark_mutated(self) -> None:
        self._mutation_revision += 1

    def _tracked_view(self, array: np.ndarray) -> np.ndarray:
        if not np.shares_memory(array, self._memmap()):
            return array
        tracked: _MutationTrackedArray = array.view(_MutationTrackedArray)
        tracked._on_mutation = self._mark_mutated
        return tracked


def _physical_shape(shape: tuple[int, ...], transposed: bool) -> tuple[int, ...]:
    if not transposed:
        return shape
    if len(shape) == 2:
        return (shape[1], shape[0])
    if len(shape) == 3:
        return (shape[1], shape[2], shape[0])
    raise ValueError("Transposed storage requires a two- or three-dimensional shape")


def _output_mapping_path(filename: str | Path | None) -> tuple[Path, bool]:
    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path, False
    temporary = tempfile.NamedTemporaryFile(prefix="eegprep-mmo-", suffix=".fdt", delete=False)
    path = Path(temporary.name)
    temporary.close()
    return path, True


def mmo(
    data_file: str | Path | None,
    dimensions: tuple[int, ...],
    writable: bool = True,
    transposed: bool = False,
    debug: bool = False,
) -> MemmapData:
    """Construct an EEGLAB-compatible memory-mapped EEG data handle.

    ``dimensions`` are always logical channel-major dimensions. Passing
    ``data_file=None`` creates a temporary zero-filled sidecar.
    """
    shape = tuple(int(item) for item in dimensions)
    if data_file is None or str(data_file) == "":
        result = MemmapData.empty(shape, transposed=transposed)
        result.debug = bool(debug)
        return result
    return MemmapData(
        data_file,
        shape,
        mode="r+" if writable else "r",
        order="F",
        transposed=transposed,
        debug=debug,
    )


def mapped_output_like(source: Any, data: Any) -> Any:
    """Keep a derived EEG data array disk-backed when its source was mapped."""
    if not isinstance(source, MemmapData) or isinstance(data, MemmapData):
        return data
    return MemmapData.from_array(
        data,
        dtype=source.dtype,
        order=source.order,
        transposed=source.transposed,
    )


class OffloadedData:
    """Handle for a dataset intentionally evicted from memory under storedisk."""

    def __init__(
        self,
        set_path: str | Path,
        shape: tuple[int, ...],
        *,
        datfile_path: str | Path | None = None,
        dtype: np.dtype | str = FDT_DTYPE,
    ) -> None:
        self.set_path = Path(set_path)
        self.datfile_path = Path(datfile_path) if datfile_path else None
        self.shape = tuple(int(item) for item in shape)
        self.dtype = np.dtype(dtype)

    @property
    def ndim(self) -> int:
        """Return the number of EEG data dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the total sample count expected on disk."""
        return int(np.prod(self.shape))

    def __array__(self, dtype: np.dtype | str | None = None, copy: bool | None = None) -> np.ndarray:
        del dtype, copy
        raise RuntimeError(
            "EEG data is offloaded to disk; retrieve the dataset with eeg_retrieve() "
            "or EEGPrepSession.retrieve() before accessing samples."
        )

    def __getitem__(self, key: Any) -> Any:
        del key
        raise RuntimeError("EEG data is offloaded to disk; retrieve the dataset before accessing samples.")

    def __copy__(self) -> "OffloadedData":
        return OffloadedData(self.set_path, self.shape, datfile_path=self.datfile_path, dtype=self.dtype)

    def __deepcopy__(self, memo: dict[int, Any]) -> "OffloadedData":
        copied = self.__copy__()
        memo[id(self)] = copied
        return copied

    def __repr__(self) -> str:
        return f"OffloadedData(set_path={str(self.set_path)!r}, shape={self.shape!r})"


def storedisk_enabled() -> bool:
    """Return whether EEGLAB-style one-resident-dataset storage is enabled."""
    return bool(int(EEG_OPTIONS.get("option_storedisk", 0) or 0))


def memmap_enabled() -> bool:
    """Return whether sidecar data should load through ``numpy.memmap``."""
    return bool(int(EEG_OPTIONS.get("option_memmapdata", 0) or 0))


def savetwofiles_enabled() -> bool:
    """Return whether ``pop_saveset`` should default to ``.set`` plus ``.fdt``."""
    return bool(int(EEG_OPTIONS.get("option_savetwofiles", 0) or 0))


def eeg_data_shape(eeg: dict[str, Any]) -> tuple[int, ...]:
    """Return the channel-major EEG data shape from metadata."""
    nbchan = int(eeg.get("nbchan", 0) or 0)
    pnts = int(eeg.get("pnts", 0) or 0)
    trials = int(eeg.get("trials", 1) or 1)
    return (nbchan, pnts, trials) if trials > 1 else (nbchan, pnts)


def write_fdt(data: Any, filename: str | Path, eeg: dict[str, Any]) -> None:
    """Write EEG data using EEGLAB's channel-fast ``.fdt`` float32 layout."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    same_backing_file = _same_backing_file(data, path)
    array = np.asarray(data, dtype=FDT_DTYPE)
    if same_backing_file:
        array = array.copy()
        _close_backing_memmap(data)
    shape = eeg_data_shape(eeg)
    if array.shape != shape:
        if array.size != int(np.prod(shape)):
            raise ValueError(f"EEG data shape {array.shape} does not match metadata shape {shape}")
        array = array.reshape(shape, order="F")
    nbchan = int(eeg.get("nbchan", 0) or 0)
    frames = int(eeg.get("pnts", 0) or 0) * int(eeg.get("trials", 1) or 1)
    matrix = array.reshape((nbchan, frames), order="F")
    np.asfortranarray(matrix).ravel(order="F").tofile(path)


def _same_backing_file(data: Any, path: Path) -> bool:
    """Return whether ``data`` maps the sidecar currently being rewritten."""
    try:
        target = path.resolve()
    except OSError:
        target = path.absolute()
    if isinstance(data, MemmapData):
        try:
            return data.path.resolve() == target
        except OSError:
            return data.path.absolute() == target
    if isinstance(data, np.memmap) and getattr(data, "filename", None):
        try:
            return Path(data.filename).resolve() == target
        except OSError:
            return Path(data.filename).absolute() == target
    return False


def _close_backing_memmap(data: Any) -> None:
    if isinstance(data, MemmapData):
        data.close()
        return
    if isinstance(data, np.memmap):
        data.flush()
        mmap_handle = getattr(data, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()


def read_fdt(filename: str | Path, eeg: dict[str, Any]) -> np.ndarray:
    """Read an EEGLAB ``.fdt`` sidecar into memory."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"EEG data sidecar not found: {path}")
    shape = eeg_data_shape(eeg)
    values = np.fromfile(path, dtype=FDT_DTYPE)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"EEG data sidecar has {values.size} samples, expected {expected}")
    return values.reshape(shape, order="F")


def memmap_fdt(filename: str | Path, eeg: dict[str, Any], *, mode: str = "r+") -> MemmapData:
    """Return a NumPy-compatible memory map for an EEGLAB ``.fdt`` sidecar."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"EEG data sidecar not found: {path}")
    return MemmapData(path, eeg_data_shape(eeg), dtype=FDT_DTYPE, mode=mode, order="F")


def dataset_set_path(eeg: dict[str, Any]) -> Path | None:
    """Return the dataset ``.set`` path from EEG metadata when available."""
    filename = _metadata_string(eeg.get("filename"))
    if not filename:
        return None
    path = Path(filename)
    if path.is_absolute():
        return path
    filepath = _metadata_string(eeg.get("filepath"))
    return Path(filepath) / path.name if filepath else path


def dataset_datfile_path(eeg: dict[str, Any]) -> Path | None:
    """Return the resolved sidecar path when EEG metadata names one."""
    data = eeg.get("data")
    if isinstance(data, MemmapData):
        return data.path
    datfile = _metadata_string(eeg.get("datfile"))
    if not datfile and isinstance(data, str) and data not in {"", "in set file"}:
        datfile = data
    if not datfile:
        return None
    path = Path(datfile)
    if path.is_absolute():
        return path
    filepath = _metadata_string(eeg.get("filepath"))
    return Path(filepath) / path.name if filepath else path


def _metadata_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def has_resident_data(eeg: dict[str, Any]) -> bool:
    """Return true when ``EEG.data`` currently occupies process memory or a map."""
    data = eeg.get("data")
    if data is None or isinstance(data, OffloadedData):
        return False
    if isinstance(data, str):
        return False
    if isinstance(data, np.ndarray):
        return data.size > 0
    if isinstance(data, list):
        return len(data) > 0
    return True


def offload_storedisk_datasets(alleeg: list[dict[str, Any]], current_indices: set[int]) -> None:
    """Replace non-current saved dataset arrays with offloaded disk handles."""
    if not storedisk_enabled():
        return
    for index, dataset in enumerate(alleeg, start=1):
        if index in current_indices or not isinstance(dataset, dict) or not dataset:
            continue
        if not has_resident_data(dataset):
            continue
        saved_state = str(dataset.get("saved") or "").lower()
        if saved_state == "justloaded":
            dataset["saved"] = "yes"
            saved_state = "yes"
        if saved_state != "yes":
            raise RuntimeError(f"Cannot offload unsaved dataset {index}; save it first.")
        set_path = dataset_set_path(dataset)
        if set_path is None or not set_path.exists():
            raise RuntimeError(f"Cannot offload dataset {index}; save it to a .set file first.")
        datfile_path = dataset_datfile_path(dataset)
        dataset["data"] = OffloadedData(
            set_path,
            eeg_data_shape(dataset),
            datfile_path=datfile_path if datfile_path and datfile_path.exists() else None,
        )


def dataset_with_loaded_data(eeg: dict[str, Any]) -> dict[str, Any]:
    """Return a loaded copy of an offloaded dataset, or a deep copy otherwise."""
    data = eeg.get("data")
    if not isinstance(data, OffloadedData):
        return deepcopy(eeg)
    from eegprep.functions.popfunc.pop_loadset import pop_loadset

    loaded = pop_loadset(str(data.set_path))
    loaded["saved"] = str(eeg.get("saved") or "yes")
    return loaded
