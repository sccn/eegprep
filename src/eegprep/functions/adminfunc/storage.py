"""Python-native storage helpers for EEGLAB-style large datasets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS

FDT_DTYPE = np.dtype("<f4")


class MemmapData:
    """NumPy-compatible handle for EEGLAB ``.fdt`` data stored on disk."""

    __array_priority__ = 1000

    def __init__(
        self,
        filename: str | Path,
        shape: tuple[int, ...],
        *,
        dtype: np.dtype | str = FDT_DTYPE,
        mode: str = "r+",
        order: str = "F",
    ) -> None:
        self.path = Path(filename)
        self._shape = tuple(int(item) for item in shape)
        self._dtype = np.dtype(dtype)
        self.mode = mode
        self.order = order
        self._array: np.memmap | None = None

    @property
    def filename(self) -> str:
        """Return the backing file path as a string."""
        return str(self.path)

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
    def T(self) -> np.ndarray:
        """Return a transposed array view."""
        return self._memmap().T

    def flush(self) -> None:
        """Flush pending writes to the backing file."""
        self._memmap().flush()

    def copy(self, order: str = "C") -> np.ndarray:
        """Return an in-memory copy of the mapped data."""
        return np.array(self._memmap(), copy=True, order=order)

    def reshape(self, *shape: Any, **kwargs: Any) -> np.ndarray:
        """Return a reshaped view using NumPy's reshape semantics."""
        return self._memmap().reshape(*shape, **kwargs)

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
        array = np.asarray(self._memmap())
        if dtype is not None:
            return array.astype(dtype, copy=bool(copy))
        if copy:
            return array.copy()
        return array

    def __getitem__(self, key: Any) -> Any:
        return self._memmap()[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._memmap()[key] = value

    def __len__(self) -> int:
        return len(self._memmap())

    def __iter__(self) -> Any:
        return iter(self._memmap())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._memmap(), name)

    def __copy__(self) -> "MemmapData":
        return MemmapData(self.path, self._shape, dtype=self._dtype, mode=self.mode, order=self.order)

    def __deepcopy__(self, memo: dict[int, Any]) -> "MemmapData":
        copied = self.__copy__()
        memo[id(self)] = copied
        return copied

    def __repr__(self) -> str:
        return f"MemmapData(path={str(self.path)!r}, shape={self._shape!r}, dtype={self._dtype})"

    def _memmap(self) -> np.memmap:
        if self._array is None:
            self._array = np.memmap(
                self.path,
                dtype=self._dtype,
                mode=self.mode,
                shape=self._shape,
                order=self.order,
            )
        return self._array


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
    array = np.asarray(data, dtype=FDT_DTYPE)
    if _same_backing_file(data, path):
        array = array.copy()
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
