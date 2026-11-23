"""Save EEGPrep STUDY structures."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from eegprep.functions.adminfunc.storage import OffloadedData
from eegprep.functions.popfunc._file_io import write_json
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.studyfunc._study_utils import build_matlab_call, ensure_study
from eegprep.functions.studyfunc.std_checkset import std_checkset


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

    The default file is an EEGLAB MATLAB v6 ``.study`` (``save -v6``).
    ``fileformat='json'`` or a ``.json`` name writes the JSON form
    ``pop_loadstudy`` still reads. ``return_com=True`` returns
    ``(study, datasets, command)`` so ``[STUDY ALLEEG] = pop_savestudy(...)``
    keeps ``ALLEEG`` as the dataset list.
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
    _write_study(path, study, fileformat)
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
    return (study, datasets, command) if return_com else study


def _write_study(path: Path, study: dict[str, Any], fileformat: str | None) -> None:
    chosen = str(fileformat or "").lower()
    if chosen not in {"", "mat", "matlab", "json"}:
        raise ValueError(
            f"pop_savestudy: format {fileformat!r} is not 'mat' or 'json'. "
            "Use fileformat='mat' for an EEGLAB .study file, or fileformat='json' for JSON."
        )
    if chosen == "" and path.suffix.lower() == ".json":
        chosen = "json"
    if chosen == "" and path.is_file() and path.read_bytes()[:1] == b"{":
        chosen = "json"
    if chosen == "json":
        write_json(path, study)
        return
    disk = _study_for_mat(study)
    temporary = path.with_name(f".{path.name}.{id(study)}.tmp")
    try:
        scipy.io.savemat(temporary, {"STUDY": disk}, appendmat=False, format="5", long_field_names=True)
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"pop_savestudy: could not write {path}: {exc}. "
            "Choose a writable .study path with File > Save current study as."
        ) from exc


def _study_for_mat(value: Any) -> Any:
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
    if isinstance(value, dict):
        if not value:
            return np.array([])
        return {str(key): _study_for_mat(item) for key, item in value.items() if isinstance(key, str)}
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
            array[0, index][key] = _study_for_mat(row.get(key, np.array([])))
    return array


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
