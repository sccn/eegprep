"""Save EEGPrep STUDY structures."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._file_io import write_json
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.studyfunc._study_utils import as_alleeg_list, build_python_call, ensure_study
from eegprep.functions.studyfunc.std_checkset import std_checkset


def pop_savestudy(
    STUDY: dict[str, Any],
    EEG: dict[str, Any] | list[dict[str, Any]] | None = None,
    *args: Any,
    filename: str | Path | None = None,
    filepath: str | Path | None = None,
    savemode: str | None = None,
    resavedatasets: str = "off",
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Save a STUDY structure as an EEGPrep JSON ``.study`` file."""
    if not STUDY:
        raise ValueError("pop_savestudy cannot save an empty STUDY")
    positional_filename, key_args = _split_filename_arg(args)
    options = parse_key_value_args(key_args, kwargs, lowercase_kwargs=True)
    filename = options.pop("filename", filename if filename is not None else positional_filename)
    filepath = options.pop("filepath", filepath)
    savemode = options.pop("savemode", savemode)
    resavedatasets = str(options.pop("resavedatasets", resavedatasets) or "off").lower()
    if options:
        raise ValueError(f"Unknown pop_savestudy option(s): {', '.join(sorted(options))}")
    if savemode not in {None, "standard", "resave", "resavegui"}:
        raise ValueError("savemode must be 'standard', 'resave', or 'resavegui'")
    if resavedatasets not in {"on", "off"}:
        raise ValueError("resavedatasets must be 'on' or 'off'")

    datasets = [deepcopy(eeg) for eeg in as_alleeg_list(EEG)] if EEG is not None else []
    study, _datasets = std_checkset(ensure_study(STUDY), datasets)
    path = _save_path(study, filename, filepath, savemode=savemode)
    path.parent.mkdir(parents=True, exist_ok=True)
    if resavedatasets == "on":
        _resave_datasets(study, datasets, path.parent)
    study["filename"] = path.name
    study["filepath"] = str(path.parent)
    study["saved"] = "yes"
    write_json(path, study)
    command = build_python_call(
        ("STUDY",),
        "pop_savestudy",
        "STUDY",
        "ALLEEG",
        filename=path.name,
        filepath=str(path.parent),
        savemode=savemode if savemode else None,
        resavedatasets="on" if resavedatasets == "on" else None,
    )
    return (study, command) if return_com else study


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
    if path.suffix.lower() != ".study":
        path = path.with_suffix(".study")
    if filepath is not None and not path.is_absolute():
        path = Path(filepath) / path.name
    return path


def _resave_datasets(study: dict[str, Any], datasets: list[dict[str, Any]], study_directory: Path) -> None:
    if not datasets:
        raise ValueError("resavedatasets='on' requires loaded ALLEEG datasets")
    infos = study.get("datasetinfo") or []
    for index, eeg in enumerate(datasets):
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
