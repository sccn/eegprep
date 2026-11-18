"""Load EEGPrep and EEGLAB STUDY structures."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from eegprep.functions.adminfunc.storage import storedisk_enabled
from eegprep.functions.popfunc._matlab_fields import convert_matlab
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.studyfunc._study_utils import build_python_call, ensure_study
from eegprep.functions.studyfunc.std_checkset import std_checkset


logger = logging.getLogger(__name__)

_LIST_FIELDS = (
    "datasetinfo",
    "cluster",
    "changrp",
    "design",
    "subject",
    "condition",
    "group",
    "session",
    "run",
    "cache",
)
_STRUCT_LIST_FIELDS = frozenset({"datasetinfo", "cluster", "changrp", "design"})
_MATLAB_HEADER = b"MATLAB"


def pop_loadstudy(
    filename: str | Path | None = None,
    *args: Any,
    filepath: str | Path | None = None,
    load_datasets: bool = True,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a STUDY saved by EEGPrep or by EEGLAB.

    EEGPrep writes a JSON ``.study`` file. EEGLAB writes a MATLAB file whose
    variable is ``STUDY``. MATLAB ``char`` arrays are ``uint16`` code points
    and are decoded as Latin-1, so bytes ``0x80``–``0xFF`` stay the characters
    EEGLAB stored. UTF-8 and UTF-16 text stored with MATLAB's Unicode types
    keeps that encoding.

    When the STUDY directory has moved, a dataset path inside the saved study
    directory is rebased onto the new directory. Other dataset locations are
    tried beside the STUDY file and under the dataset's own parent folder
    names. The study directory itself is not shortened. ``option_storedisk``
    selects header-only or full data loading. An empty ``changrp`` is left
    empty. Measure arrays saved in the STUDY stay in place. EEGLAB clears
    those in-memory copies on load because it reads them back from separate
    files.

    Args:
        filename: STUDY file name or path.
        filepath: Directory used when ``filename`` is not absolute.
        load_datasets: When false, return the STUDY without opening datasets.
        return_com: Also return the history command.

    Returns:
        ``(STUDY, ALLEEG)`` or ``(STUDY, ALLEEG, command)``.
    """
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    filename = options.pop("filename", filename)
    filepath = options.pop("filepath", filepath)
    load_datasets = is_on(options.pop("load_datasets", load_datasets))
    if options:
        raise ValueError(f"Unknown pop_loadstudy option(s): {', '.join(sorted(options))}")
    if filename is None:
        raise ValueError("pop_loadstudy requires a filename")

    path = _study_path(filename, filepath)
    study = ensure_study(_read_study(path))
    old_filepath = _text(study.get("filepath"))
    study["filename"] = path.name
    study["filepath"] = str(path.parent)
    study["etc"]["oldfilepath"] = old_filepath
    _warn_legacy_design(study)
    alleeg = _load_datasets(study) if load_datasets else []
    study, alleeg = std_checkset(study, alleeg)
    study["saved"] = "yes"
    command_kwargs: dict[str, Any] = {"filename": path.name, "filepath": str(path.parent)}
    if not load_datasets:
        command_kwargs["load_datasets"] = False
    command = build_python_call(("STUDY", "ALLEEG"), "pop_loadstudy", **command_kwargs)
    return (study, alleeg, command) if return_com else (study, alleeg)


def _study_path(filename: str | Path, filepath: str | Path | None) -> Path:
    path = Path(filename)
    if filepath is not None and not path.is_absolute():
        path = Path(filepath) / path
    return path


def _read_study(path: Path) -> dict[str, Any]:
    header = path.read_bytes()[: len(_MATLAB_HEADER)]
    if header.startswith(_MATLAB_HEADER):
        return _study_from_mat(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("STUDY file must contain a JSON object")
    return payload


def _study_from_mat(path: Path) -> dict[str, Any]:
    loaded = scipy.io.loadmat(
        path,
        struct_as_record=False,
        squeeze_me=False,
        appendmat=False,
        uint16_codec="latin-1",
    )
    if "STUDY" not in loaded:
        raise ValueError(f"{path} does not contain a STUDY variable")
    study = convert_matlab(loaded["STUDY"], "STUDY")
    if isinstance(study, list):
        study = study[0] if study else {}
    if not isinstance(study, dict):
        raise ValueError(f"{path} does not contain a STUDY structure")
    return _normalize_loaded_study(study)


def _normalize_loaded_study(study: dict[str, Any]) -> dict[str, Any]:
    for field in _LIST_FIELDS:
        if field in study:
            study[field] = _as_list(study[field], wrap_dict=field in _STRUCT_LIST_FIELDS)
    for field in ("notes", "history", "filename", "filepath", "name", "task", "saved"):
        if field in study:
            study[field] = _text(study[field])
    designs = study.get("design")
    if isinstance(designs, list):
        for design in designs:
            if not isinstance(design, dict):
                continue
            for field in ("variable", "cell"):
                if field in design:
                    design[field] = _as_list(design[field], wrap_dict=True)
            include = design.get("include")
            if isinstance(include, np.ndarray) and include.size == 0:
                design["include"] = []
    return study


def _as_list(value: Any, *, wrap_dict: bool) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        if value.dtype.names or value.dtype == object:
            value = value.reshape(-1, order="F").tolist()
        else:
            return value.reshape(-1, order="F").tolist()
    if isinstance(value, dict):
        return [value] if wrap_dict else value
    if value is None:
        return []
    if isinstance(value, tuple):
        return list(value)
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("latin-1")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.dtype.kind in "SU":
            item = value.reshape(-1)[0]
            if isinstance(item, bytes):
                return item.decode("latin-1")
            return str(item)
        return str(value.reshape(-1)[0])
    return str(value)


def _warn_legacy_design(study: dict[str, Any]) -> None:
    designs = study.get("design") or []
    if isinstance(designs, list) and any(isinstance(design, dict) and "cell" in design for design in designs):
        logger.warning(
            "This STUDY stores design.cell from an older EEGLAB. "
            "Recompute ERP, ERSP, ITC, and spectrum measures before using them."
        )


def _load_datasets(study: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = []
    missing = []
    loadmode = "info" if storedisk_enabled() else "all"
    for index, info in enumerate(study.get("datasetinfo") or [], start=1):
        if not isinstance(info, dict):
            missing.append(f"#{index}: datasetinfo entry is not a structure")
            continue
        candidates = _dataset_candidates(study, info)
        if not candidates:
            missing.append(f"#{index}: no dataset filename")
            continue
        found = next((candidate for candidate in candidates if candidate.is_file()), None)
        if found is None:
            missing.append(f"#{index}: {candidates[0]}")
            continue
        info["filename"] = found.name
        info["filepath"] = str(found.parent)
        datasets.append(pop_loadset(str(found), loadmode=loadmode))
    if missing:
        raise FileNotFoundError(
            "pop_loadstudy could not load all STUDY datasets; missing "
            + ", ".join(missing)
            + ". Move the dataset files back, or call pop_loadstudy(..., load_datasets=False)."
        )
    return datasets


def _dataset_candidates(study: dict[str, Any], info: dict[str, Any]) -> list[Path]:
    filename = _text(info.get("filename"))
    if not filename:
        return []
    stored = _portable_path(info.get("filepath"))
    new_root = Path(_text(study.get("filepath")))
    old_root = _portable_path(study.get("etc", {}).get("oldfilepath"))
    dataset_dir = stored
    if str(old_root) and str(old_root) != str(new_root):
        try:
            dataset_dir = new_root / stored.relative_to(old_root)
        except ValueError:
            dataset_dir = stored
    sub1 = stored.name
    sub2 = stored.parent.name
    ordered = [
        dataset_dir / filename,
        stored / filename if str(stored) else None,
        new_root / filename,
        new_root / sub1 / filename if sub1 not in {"", "."} else None,
        new_root / sub2 / filename if sub2 not in {"", ".", "/"} else None,
        new_root / sub2 / sub1 / filename if sub1 not in {"", "."} and sub2 not in {"", ".", "/"} else None,
        Path(str(stored).lower()) / filename.lower() if str(stored) else None,
        Path.cwd() / filename.lower(),
    ]
    candidates = []
    seen: set[str] = set()
    for candidate in ordered:
        if candidate is None:
            continue
        key = os.path.normcase(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _portable_path(value: Any) -> Path:
    text = _text(value)
    if os.name != "nt" and len(text) >= 2 and text[1] == ":":
        text = text[2:]
    if os.name != "nt":
        text = text.replace("\\", "/")
    return Path(text)


__all__ = ["pop_loadstudy"]
