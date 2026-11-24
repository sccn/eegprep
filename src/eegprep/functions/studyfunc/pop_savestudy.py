"""Save EEGPrep STUDY structures."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io

from eegprep.functions.adminfunc.storage import OffloadedData
from eegprep.functions.popfunc._file_io import write_json
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.studyfunc._study_utils import build_matlab_call, ensure_study
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.studyfunc.std_rmalldatafields import std_rmalldatafields


def pop_savestudy(
    STUDY: dict[str, Any],
    EEG: dict[str, Any] | list[dict[str, Any]] | None = None,
    *args: Any,
    filename: str | Path | None = None,
    filepath: str | Path | None = None,
    savemode: str | None = None,
    resavedatasets: str = "off",
    fileformat: str | None = None,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Save a STUDY structure.

    The default file is an EEGLAB MATLAB v6 ``.study`` (``save -v6``). A file
    larger than the v6 limit is written as MATLAB v7.3, the same fallback as
    ``pop_savestudy.m``. Precomputed measure arrays are removed from the file
    and kept on the returned STUDY. ``fileformat='json'`` writes JSON;
    resaving a ``.study`` that currently holds JSON writes MATLAB, which is
    what EEGLAB can load. The return is always ``(study, datasets, command)``.
    """
    if not STUDY:
        raise ValueError("pop_savestudy cannot save an empty STUDY")
    positional_filename, key_args = _split_filename_arg(args)
    options = parse_key_value_args(key_args, kwargs, lowercase_kwargs=True)
    filename = options.pop("filename", filename if filename is not None else positional_filename)
    filepath = options.pop("filepath", filepath)
    savemode = options.pop("savemode", savemode)
    resavedatasets = str(options.pop("resavedatasets", resavedatasets) or "off").lower()
    fileformat = options.pop("format", options.pop("fileformat", fileformat))
    if options:
        raise ValueError(f"Unknown pop_savestudy option(s): {', '.join(sorted(options))}")
    if savemode not in {None, "standard", "resave", "resavegui"}:
        raise ValueError("savemode must be 'standard', 'resave', or 'resavegui'")
    if resavedatasets not in {"on", "off"}:
        raise ValueError("resavedatasets must be 'on' or 'off'")

    study, datasets = std_checkset(ensure_study(STUDY), EEG)
    datasets = [deepcopy(eeg) for eeg in datasets]
    path = _save_path(study, filename, filepath, savemode=savemode)
    path.parent.mkdir(parents=True, exist_ok=True)
    if resavedatasets == "on":
        _resave_datasets(study, datasets, path.parent)
    study["filename"] = path.name
    study["filepath"] = str(path.parent)
    study["saved"] = "yes"
    _write_study(path, std_rmalldatafields(study), fileformat, savemode=savemode)
    command = build_matlab_call(
        ("STUDY", "ALLEEG"),
        "pop_savestudy",
        "STUDY",
        "ALLEEG",
        filename=path.name,
        filepath=str(path.parent),
        savemode=savemode if savemode else None,
        resavedatasets="on" if resavedatasets == "on" else None,
    )
    return study, datasets, command


def _write_study(path: Path, study: dict[str, Any], fileformat: str | None, *, savemode: str | None) -> None:
    chosen = str(fileformat or "").lower()
    if chosen not in {"", "mat", "matlab", "json"}:
        raise ValueError(
            f"pop_savestudy: format {fileformat!r} is not 'mat' or 'json'. "
            "Use fileformat='mat' for an EEGLAB .study file, or fileformat='json' for JSON."
        )
    if savemode == "resave":
        chosen = "mat"
    elif chosen == "" and path.suffix.lower() == ".json":
        chosen = "json"
    if chosen == "json":
        write_json(path, study)
        return
    disk = _study_for_mat(study)
    temporary = path.with_name(f".{path.name}.{id(study)}.tmp")
    try:
        scipy.io.savemat(temporary, {"STUDY": disk}, appendmat=False, format="5", long_field_names=True)
        temporary.replace(path)
    except (OverflowError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        if not _mat5_too_large(exc):
            raise ValueError(
                f"pop_savestudy: could not write {path}: {exc}. "
                "Choose a writable .study path with File > Save current study as."
            ) from exc
        _write_study_v73(path, disk)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"pop_savestudy: could not write {path}: {exc}. "
            "Choose a writable .study path with File > Save current study as."
        ) from exc


def _mat5_too_large(exc: BaseException) -> bool:
    if isinstance(exc, OverflowError):
        return True
    text = str(exc).lower()
    return "2gb" in text or "2 gb" in text or "too large" in text or "too big" in text


class _Mat73Writer:
    def __init__(self) -> None:
        self.count = 0


def _mat73_header() -> bytes:
    """128-byte MATLAB v7.3 header: description, subsystem offset, version, endian.

    v7.3 is version ``0x0200``. Little-endian bytes 124-125 are ``00 02``.
    """
    text = b"MATLAB 7.3 MAT-file, Platform: eegprep, Created by pop_savestudy"
    return text[:116].ljust(116, b" ") + (0).to_bytes(8, "little") + (0x0200).to_bytes(2, "little") + b"IM"


def _write_study_v73(path: Path, study: Any) -> None:
    temporary = path.with_name(f".{path.name}.v73.tmp")
    temporary.unlink(missing_ok=True)
    writer = _Mat73Writer()
    with h5py.File(temporary, "w", userblock_size=512) as handle:
        _write_mat73_value(handle, "STUDY", study, handle, writer)
    with temporary.open("r+b") as stream:
        stream.write(_mat73_header())
    temporary.replace(path)


def _write_mat73_value(parent: h5py.Group, name: str, value: Any, root: h5py.File, writer: _Mat73Writer) -> None:
    if isinstance(value, dict):
        _write_mat73_mapping(parent, name, value, root, writer)
        return
    if isinstance(value, str):
        _write_mat73_char(parent, name, value)
        return
    if isinstance(value, (bool, np.bool_)):
        dataset = parent.create_dataset(name, data=np.uint8([[1 if value else 0]]))
        _set_matlab_class(dataset, "logical")
        return
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        dataset = parent.create_dataset(name, data=np.array([[float(value)]], dtype=np.float64))
        _set_matlab_class(dataset, "double")
        return
    if isinstance(value, np.ndarray) and value.dtype.names:
        _write_mat73_struct(parent, name, value, root, writer)
        return
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.size == 0:
            _write_mat73_empty(parent, name, value.shape, "cell")
            return
        _write_mat73_cell(parent, name, value, root, writer)
        return
    if isinstance(value, np.ndarray):
        if value.size == 0:
            _write_mat73_empty(parent, name, value.shape, "double")
            return
        dataset = parent.create_dataset(name, data=np.asarray(value, dtype=np.float64).T)
        _set_matlab_class(dataset, "double")
        return
    _write_mat73_char(parent, name, str(value))


def _set_matlab_class(node: h5py.Dataset | h5py.Group, class_name: str) -> None:
    """Store ``MATLAB_class`` as a fixed-length ASCII scalar, not a vlen string."""
    node.attrs["MATLAB_class"] = np.bytes_(class_name)


def _write_mat73_empty(parent: h5py.Group, name: str, shape: tuple[int, ...], class_name: str) -> None:
    """Write an empty MATLAB array as uint64 dimensions plus ``MATLAB_empty``.

    A 1-D empty is MATLAB ``[]``, stored as 0-by-0. An empty char has no
    ``MATLAB_int_decode`` attribute.
    """
    dims = shape if len(shape) >= 2 else (0, 0)
    dataset = parent.create_dataset(name, data=np.asarray(dims, dtype=np.uint64).reshape(-1))
    _set_matlab_class(dataset, class_name)
    dataset.attrs["MATLAB_empty"] = np.uint8(1)


def _write_mat73_char(parent: h5py.Group, name: str, text: str) -> None:
    if not text:
        _write_mat73_empty(parent, name, (0, 0), "char")
        return
    codes = np.frombuffer(text.encode("utf-16-le"), dtype="<u2").reshape(-1, 1)
    dataset = parent.create_dataset(name, data=codes)
    _set_matlab_class(dataset, "char")
    dataset.attrs["MATLAB_int_decode"] = np.int32(2)


def _referenced(root: h5py.File, writer: _Mat73Writer, value: Any) -> h5py.Reference:
    """Store ``value`` directly in ``#refs#`` and return its object reference.

    MATLAB follows the reference to the dataset or group itself. A wrapper
    group whose only child is ``value`` is not a v7.3 object.
    """
    refs = root.require_group("#refs#")
    writer.count += 1
    name = f"r{writer.count}"
    _write_mat73_value(refs, name, value, root, writer)
    return refs[name].ref


def _write_mat73_mapping(
    parent: h5py.Group,
    name: str,
    mapping: dict[str, Any],
    root: h5py.File,
    writer: _Mat73Writer,
) -> None:
    group = parent.create_group(name)
    _set_matlab_class(group, "struct")
    for key, item in mapping.items():
        _write_mat73_value(group, str(key), item, root, writer)


def _write_mat73_struct(
    parent: h5py.Group,
    name: str,
    array: np.ndarray,
    root: h5py.File,
    writer: _Mat73Writer,
) -> None:
    group = parent.create_group(name)
    _set_matlab_class(group, "struct")
    rows = array.reshape(-1, order="F")
    ref_dtype = h5py.ref_dtype
    for field in array.dtype.names or ():
        pointers = group.create_dataset(field, (len(rows), 1), dtype=ref_dtype)
        for index, row in enumerate(rows):
            pointers[index, 0] = _referenced(root, writer, row[field])


def _write_mat73_cell(
    parent: h5py.Group,
    name: str,
    array: np.ndarray,
    root: h5py.File,
    writer: _Mat73Writer,
) -> None:
    flat = array.reshape(-1, order="F")
    refs = np.empty(array.T.shape, dtype=h5py.ref_dtype)
    for index, item in enumerate(flat):
        refs.reshape(-1, order="F")[index] = _referenced(root, writer, item)
    dataset = parent.create_dataset(name, data=refs)
    _set_matlab_class(dataset, "cell")


def _study_for_mat(value: Any, *, key: str | None = None) -> Any:
    if value is None:
        return np.array([])
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return np.array([]) if not np.isfinite(number) else number
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if key == "sets":
        return _sets_matrix(value)
    if isinstance(value, dict):
        if not value:
            return np.array([])
        return {str(name): _study_for_mat(item, key=str(name)) for name, item in value.items() if isinstance(name, str)}
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return value
        if value.size == 0:
            return np.array([])
        if value.dtype == object or value.dtype.kind not in "iufbSU":
            return _study_for_mat(value.reshape(-1, order="F").tolist())
        if value.dtype.kind in "iu":
            return value.astype(np.float64)
        return value
    if isinstance(value, (list, tuple)):
        items = list(value)
        if not items:
            return np.array([])
        if all(isinstance(item, dict) for item in items):
            return _struct_rows(items)
        if all(isinstance(item, str) for item in items):
            cell = np.empty((1, len(items)), dtype=object)
            for index, item in enumerate(items):
                cell[0, index] = item
            return cell
        if all(
            isinstance(item, (int, float, np.integer, np.floating)) and not isinstance(item, bool) for item in items
        ):
            return np.asarray([float(item) for item in items], dtype=float).reshape(1, -1)
        cell = np.empty((1, len(items)), dtype=object)
        for index, item in enumerate(items):
            cell[0, index] = _study_for_mat(item)
        return cell
    return str(value)


def _struct_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if isinstance(key, str) and key not in seen:
                seen.add(key)
                keys.append(key)
    if not keys:
        return np.array([])
    array = np.empty((1, len(rows)), dtype=[(key, object) for key in keys])
    for index, row in enumerate(rows):
        for key in keys:
            array[0, index][key] = _study_for_mat(row.get(key, np.array([])), key=key)
    return array


def _sets_matrix(value: Any) -> np.ndarray:
    """Return ``cluster.sets`` as a numeric rows-by-components matrix.

    EEGLAB's ``std_checkset`` calls ``max(sets(:))``. A cell of row vectors is
    rejected there as an invalid data type.
    """
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.zeros((0, 0), dtype=float)
        array = np.asarray(value, dtype=float)
        if array.ndim == 1:
            return array.reshape(-1, 1)
        return array
    rows = list(value)
    if not rows:
        return np.zeros((0, 0), dtype=float)
    if all(isinstance(item, (int, float, np.integer, np.floating)) and not isinstance(item, bool) for item in rows):
        return np.asarray([float(item) for item in rows], dtype=float).reshape(-1, 1)
    matrix = []
    for row in rows:
        if isinstance(row, np.ndarray):
            row = row.reshape(-1).tolist()
        if not isinstance(row, (list, tuple)):
            raise ValueError(
                "pop_savestudy: cluster.sets must be a numeric matrix (rows x components). "
                "Pass a 2-D array of dataset indices, then save the study again."
            )
        matrix.append([float(item) for item in row])
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(
            "pop_savestudy: cluster.sets rows have different lengths. "
            "Use one column per component and the same number of columns in every row, then save again."
        )
    return np.asarray(matrix, dtype=float)


def _split_filename_arg(args: tuple[Any, ...]) -> tuple[Any, tuple[Any, ...]]:
    if len(args) == 1:
        return args[0], ()
    return None, args


def _save_path(
    study: dict[str, Any],
    filename: str | Path | None,
    filepath: str | Path | None,
    *,
    savemode: str | None,
) -> Path:
    if savemode == "resave":
        filename = study.get("filename") or filename
        filepath = study.get("filepath") or filepath
    if filename is None:
        filename = study.get("filename")
    if filename is None or str(filename) == "":
        raise ValueError("File name required to save the study")
    path = Path(filename)
    if path.suffix.lower() not in {".study", ".json"}:
        path = path.with_suffix(".study")
    if filepath is not None and not path.is_absolute():
        path = Path(filepath) / path.name
    return path


def _resave_datasets(study: dict[str, Any], datasets: list[dict[str, Any]], study_directory: Path) -> None:
    if not datasets:
        raise ValueError("resavedatasets='on' requires loaded ALLEEG datasets")
    infos = study.get("datasetinfo") or []
    for index, eeg in enumerate(datasets):
        if isinstance(eeg.get("data"), OffloadedData):
            eeg = pop_loadset(str(eeg["data"].set_path))
            datasets[index] = eeg
        info = infos[index] if index < len(infos) and isinstance(infos[index], dict) else {}
        filename = eeg.get("filename") or info.get("filename")
        if not filename:
            raise ValueError(f"Dataset {index + 1} needs a filename before resavedatasets='on'")
        filepath = eeg.get("filepath") or info.get("filepath") or study_directory
        path = Path(str(filename))
        if not path.is_absolute():
            path = Path(filepath) / path.name
        pop_saveset(eeg, path)
        eeg["filename"] = path.name
        eeg["filepath"] = str(path.parent)
        eeg["saved"] = "yes"


__all__ = ["pop_savestudy"]
