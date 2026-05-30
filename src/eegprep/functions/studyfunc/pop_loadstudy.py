"""Load EEGPrep STUDY structures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.studyfunc._study_utils import build_python_call, dataset_path, ensure_study
from eegprep.functions.studyfunc.std_checkset import std_checkset


def pop_loadstudy(
    filename: str | Path | None = None,
    *args: Any,
    filepath: str | Path | None = None,
    load_datasets: bool = True,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a STUDY JSON file saved by ``pop_savestudy``."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    filename = options.pop("filename", filename)
    filepath = options.pop("filepath", filepath)
    load_datasets = bool(options.pop("load_datasets", load_datasets))
    if options:
        raise ValueError(f"Unknown pop_loadstudy option(s): {', '.join(sorted(options))}")
    if filename is None:
        raise ValueError("pop_loadstudy requires a filename")

    path = _study_path(filename, filepath)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("STUDY file must contain a JSON object")
    study = ensure_study(payload)
    old_filepath = study.get("filepath", "")
    study["filename"] = path.name
    study["filepath"] = str(path.parent)
    study["etc"].setdefault("oldfilepath", old_filepath)
    alleeg = _load_datasets(study) if load_datasets else []
    study, alleeg = std_checkset(study, alleeg)
    study["saved"] = "yes"
    command = build_python_call(
        ("STUDY", "ALLEEG"),
        "pop_loadstudy",
        filename=path.name,
        filepath=str(path.parent),
    )
    return (study, alleeg, command) if return_com else (study, alleeg)


def _study_path(filename: str | Path, filepath: str | Path | None) -> Path:
    path = Path(filename)
    if filepath is not None and not path.is_absolute():
        path = Path(filepath) / path
    return path


def _load_datasets(study: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = []
    missing = []
    for index, info in enumerate(study.get("datasetinfo") or [], start=1):
        path = _resolved_dataset_path(study, info)
        if path is None:
            missing.append(f"#{index}: no dataset filename")
            continue
        if not path.exists():
            missing.append(f"#{index}: {path}")
            continue
        datasets.append(pop_loadset(str(path)))
    if missing:
        raise FileNotFoundError(
            "pop_loadstudy could not load all STUDY datasets; missing "
            + ", ".join(missing)
            + ". Move the dataset files back, or call pop_loadstudy(..., load_datasets=False)."
        )
    return datasets


def _resolved_dataset_path(study: dict[str, Any], info: dict[str, Any]) -> Path | None:
    path = dataset_path(info)
    if path is None:
        return None
    if path.is_absolute() or path.exists():
        return path
    study_path = Path(study.get("filepath") or "")
    return study_path / path


__all__ = ["pop_loadstudy"]
