"""Shared STUDY metadata helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.popfunc._file_io import channel_labels, json_safe
from eegprep.functions.popfunc._plot_utils import python_literal
from eegprep.functions.popfunc._pop_utils import parse_numeric_sequence, parse_text_tokens


DATASETINFO_FIELDS = ("index", "setname", "filename", "filepath", "subject", "condition", "group", "session", "run")
SUMMARY_FIELDS = ("subject", "condition", "group", "session", "run")
RESERVED_VARIABLE_FIELDS = {
    "index",
    "setname",
    "filename",
    "filepath",
    "comps",
    "trialinfo",
}
STUDY_DATA_FIELDS = (
    "erpdata",
    "erptimes",
    "specdata",
    "specfreqs",
    "erspdata",
    "ersptimes",
    "erspfreqs",
    "erspdatatrials",
    "erspsubjinds",
    "erspbase",
    "ersptrialinfo",
    "itcdata",
    "itcfreqs",
    "itctimes",
    "erpimdata",
    "erpimevents",
    "erpimtrials",
    "erpimtimes",
)


def as_alleeg_list(ALLEEG: Any) -> list[dict[str, Any]]:
    """Return ``ALLEEG`` as a list of EEG dictionaries."""
    if ALLEEG is None:
        return []
    if isinstance(ALLEEG, dict):
        return [ALLEEG]
    if not isinstance(ALLEEG, list) or not all(isinstance(item, dict) for item in ALLEEG):
        raise TypeError("ALLEEG must be a list of EEG dataset dictionaries")
    return list(ALLEEG)


def ensure_study(STUDY: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a STUDY dict with EEGLAB-style base fields."""
    study = deepcopy(STUDY) if isinstance(STUDY, dict) else {}
    study.setdefault("name", "")
    study.setdefault("task", "")
    study.setdefault("notes", "")
    study.setdefault("filename", "")
    study.setdefault("filepath", "")
    study.setdefault("datasetinfo", [])
    study.setdefault("subject", [])
    study.setdefault("condition", [])
    study.setdefault("group", [])
    study.setdefault("session", [])
    study.setdefault("run", [])
    study.setdefault("changrp", [])
    study.setdefault("cluster", [{"name": "ParentCluster", "sets": [], "comps": [], "child": []}])
    study.setdefault("design", [])
    study.setdefault("currentdesign", 0)
    study.setdefault("cache", [])
    study.setdefault("preclust", _default_preclust())
    study.setdefault("history", "STUDY = []")
    study.setdefault("saved", "no")
    etc = study.get("etc")
    if not isinstance(etc, dict):
        etc = {}
    etc.setdefault("version", "")
    etc.setdefault("warnmemory", 1)
    etc.setdefault("eegprep", {})
    study["etc"] = etc
    return study


def sync_datasetinfo(study: dict[str, Any], datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Synchronize STUDY.datasetinfo with loaded datasets while preserving metadata edits."""
    study = ensure_study(study)
    existing = _datasetinfo_list(study.get("datasetinfo"))
    infos: list[dict[str, Any]] = []
    for index, eeg in enumerate(datasets, start=1):
        previous = existing[index - 1] if index <= len(existing) else {}
        info = dataset_info_from_eeg(index, eeg)
        for field in ("subject", "condition", "group", "session", "run", "comps"):
            if not _empty_value(previous.get(field)):
                info[field] = previous[field]
        for key, value in previous.items():
            if key not in info:
                info[key] = deepcopy(value)
        infos.append(info)
    if not datasets:
        infos = existing
        for index, info in enumerate(infos, start=1):
            info["index"] = index
    study["datasetinfo"] = infos
    refresh_study_summaries(study)
    return study


def dataset_info_from_eeg(index: int, eeg: dict[str, Any]) -> dict[str, Any]:
    """Build one STUDY.datasetinfo entry from an EEG dict."""
    subject = eeg.get("subject")
    info = {
        "index": int(index),
        "setname": str(eeg.get("setname") or ""),
        "filename": str(eeg.get("filename") or ""),
        "filepath": str(eeg.get("filepath") or ""),
        "subject": str(subject) if not _empty_value(subject) else f"S{index}",
        "condition": "" if _empty_value(eeg.get("condition")) else str(eeg.get("condition")),
        "group": "" if _empty_value(eeg.get("group")) else str(eeg.get("group")),
        "session": _normalize_optional_number(eeg.get("session")),
        "run": _normalize_optional_number(eeg.get("run")),
        "comps": _default_components(eeg),
    }
    if not _empty_value(eeg.get("task")):
        info["task"] = eeg.get("task")
    trialinfo = trialinfo_from_eeg(eeg)
    if trialinfo:
        info["trialinfo"] = trialinfo
    return info


def refresh_study_summaries(study: dict[str, Any]) -> None:
    """Refresh STUDY.subject/group/condition/session/run summary fields."""
    infos = _datasetinfo_list(study.get("datasetinfo"))
    for field in SUMMARY_FIELDS:
        study[field] = _unique_eeglab_values(info.get(field) for info in infos)


def check_datasetinfo(study: dict[str, Any], datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Return deterministic STUDY/dataset consistency details."""
    infos = _datasetinfo_list(study.get("datasetinfo"))
    result: dict[str, Any] = {
        "dataset_count": len(datasets),
        "datasetinfo_count": len(infos),
        "dataset_count_match": len(datasets) == len(infos),
        "filename_mismatches": [],
        "missing_files": [],
        "same_srate": True,
        "same_time_grid": True,
        "same_channel_count": True,
        "same_channel_labels": True,
        "same_epoch_count": True,
        "problems": [],
    }
    if len(datasets) != len(infos):
        result["problems"].append("datasetinfo count does not match ALLEEG")

    srates = [_float_or_none(eeg.get("srate")) for eeg in datasets]
    grids = [
        (
            int(eeg.get("pnts", 0) or 0),
            _float_or_none(eeg.get("srate")),
            _float_or_none(eeg.get("xmin")),
            _float_or_none(eeg.get("xmax")),
        )
        for eeg in datasets
    ]
    channel_counts = [int(eeg.get("nbchan", 0) or 0) for eeg in datasets]
    epoch_counts = [int(eeg.get("trials", 1) or 1) for eeg in datasets]
    labels = [channel_labels(eeg) for eeg in datasets]
    result["sampling_rates"] = _unique_values(srates)
    result["time_grids"] = _unique_values(grids)
    result["channel_counts"] = _unique_values(channel_counts)
    result["epoch_counts"] = _unique_values(epoch_counts)
    result["same_srate"] = len(result["sampling_rates"]) <= 1
    result["same_time_grid"] = len(result["time_grids"]) <= 1
    result["same_channel_count"] = len(result["channel_counts"]) <= 1
    result["same_epoch_count"] = len(result["epoch_counts"]) <= 1
    result["same_channel_labels"] = len({tuple(item) for item in labels}) <= 1
    for key, message in (
        ("same_srate", "datasets do not share one sampling rate"),
        ("same_time_grid", "datasets do not share one time grid"),
        ("same_channel_count", "datasets do not share one channel count"),
        ("same_channel_labels", "datasets do not share one channel-label set"),
        ("same_epoch_count", "datasets do not share one epoch count"),
    ):
        if not result[key]:
            result["problems"].append(message)

    for index, (info, eeg) in enumerate(zip(infos, datasets), start=1):
        expected = (str(eeg.get("filename") or ""), str(eeg.get("filepath") or ""))
        actual = (str(info.get("filename") or ""), str(info.get("filepath") or ""))
        if expected != actual:
            result["filename_mismatches"].append({"index": index, "study": actual, "alleeg": expected})
        path = dataset_path(info)
        if path is not None and not path.exists():
            result["missing_files"].append({"index": index, "path": str(path)})
    if result["filename_mismatches"]:
        result["problems"].append("datasetinfo filename/path does not match ALLEEG")
    if result["missing_files"]:
        result["problems"].append("one or more STUDY dataset files are missing")
    return json_safe(result)


def store_consistency(study: dict[str, Any], datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Store dataset consistency checks under STUDY.etc.eegprep."""
    study = ensure_study(study)
    study["etc"].setdefault("eegprep", {})["dataset_consistency"] = check_datasetinfo(study, datasets)
    return study


def dataset_path(info: dict[str, Any]) -> Path | None:
    """Return a dataset path from STUDY.datasetinfo when one is available."""
    filename = str(info.get("filename") or "")
    if not filename:
        return None
    path = Path(filename)
    if path.is_absolute():
        return path
    filepath = str(info.get("filepath") or "")
    return Path(filepath) / filename if filepath else path


def trialinfo_from_eeg(eeg: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract simple trial metadata from an epoched EEG dict."""
    trials = int(eeg.get("trials", 1) or 1)
    if trials <= 1:
        return []
    epoch = eeg.get("epoch")
    if epoch is None:
        epoch = []
    if isinstance(epoch, np.ndarray):
        epoch = epoch.tolist()
    if isinstance(epoch, list) and len(epoch) == trials:
        rows = []
        for entry in epoch:
            if not isinstance(entry, dict):
                rows.append({})
                continue
            row = {
                key: deepcopy(value)
                for key, value in entry.items()
                if key not in {"event", "eventtype", "eventlatency", "eventduration"} and not _empty_value(value)
            }
            rows.append(row)
        if any(rows):
            return rows
    return []


def available_variables(study: dict[str, Any]) -> list[str]:
    """Return factor names available from datasetinfo and trialinfo."""
    names: list[str] = []
    for info in _datasetinfo_list(study.get("datasetinfo")):
        for key, value in info.items():
            if key not in RESERVED_VARIABLE_FIELDS and not _empty_value(value) and key not in names:
                names.append(key)
        for row in info.get("trialinfo") or []:
            if isinstance(row, dict):
                for key, value in row.items():
                    if not _empty_value(value) and key not in names:
                        names.append(key)
    preferred = [field for field in ("condition", "group", "session", "run", "subject") if field in names]
    return preferred + sorted(name for name in names if name not in preferred)


def variable_values(study: dict[str, Any], label: str) -> list[Any]:
    """Return unique values for one STUDY variable label."""
    values: list[Any] = []
    for info in _datasetinfo_list(study.get("datasetinfo")):
        if label in info and not _empty_value(info.get(label)):
            values.append(info[label])
        for row in info.get("trialinfo") or []:
            if isinstance(row, dict) and label in row and not _empty_value(row.get(label)):
                values.append(row[label])
    return _unique_eeglab_values(values)


def parse_design_values(value: Any) -> list[Any]:
    """Parse user-entered STUDY design values."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [item for item in value if not _empty_value(item)]
    text = str(value).strip()
    if not text or text in {"[]", "{}"}:
        return []
    try:
        return parse_numeric_sequence(text, dtype=int)
    except ValueError:
        return parse_text_tokens(text, parse_ints=True)


def parse_optional_int_text(value: Any) -> Any:
    """Parse optional 1-based session/run metadata from GUI text."""
    if _empty_value(value):
        return []
    values = parse_numeric_sequence(value, dtype=int)
    if len(values) == 1:
        return int(values[0])
    return values


def build_python_call(targets: tuple[str, ...], function_name: str, *args: str, **kwargs: Any) -> str:
    """Build a pasteable Python history command."""
    pieces = list(args)
    pieces.extend(f"{key}={python_literal(value)}" for key, value in kwargs.items() if not _empty_value(value))
    assignment = ", ".join(targets)
    return f"{assignment} = {function_name}({', '.join(pieces)})"


def clear_study_data_fields(study: dict[str, Any]) -> dict[str, Any]:
    """Remove precomputed measure arrays from STUDY channel/component groups."""
    study = deepcopy(study)
    for field in ("changrp", "cluster"):
        value = study.get(field)
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    for data_field in STUDY_DATA_FIELDS:
                        entry.pop(data_field, None)
        elif isinstance(value, dict):
            for data_field in STUDY_DATA_FIELDS:
                value.pop(data_field, None)
    return study


def _datasetinfo_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [deepcopy(value)]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    return [deepcopy(item) for item in value if isinstance(item, dict)]


def _default_preclust() -> dict[str, list[Any]]:
    return {
        "erpclusttimes": [],
        "specclustfreqs": [],
        "erspclustfreqs": [],
        "erspclusttimes": [],
    }


def _default_components(eeg: dict[str, Any]) -> list[int]:
    weights = eeg.get("icaweights")
    if weights is None:
        return []
    array = np.asarray(weights)
    if array.size == 0 or array.ndim < 2:
        return []
    return list(range(1, int(array.shape[0]) + 1))


def _normalize_optional_number(value: Any) -> Any:
    if _empty_value(value):
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return [_normalize_optional_number(item) for item in value if not _empty_value(item)]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return int(value) if float(value).is_integer() else float(value)
    return value


def _unique_values(values: Any) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if _empty_value(value):
            continue
        key = repr(json_safe(value))
        if key in seen:
            continue
        seen.add(key)
        unique.append(json_safe(value))
    return unique


def _unique_eeglab_values(values: Any) -> list[Any]:
    unique = _unique_values(values)
    try:
        return sorted(unique)
    except TypeError:
        return unique


def _empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _float_or_none(value: Any) -> float | None:
    if _empty_value(value):
        return None
    return float(value)
