"""Remove STUDY datasets by metadata and dataset-shape criteria."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.studyfunc._study_utils import (
    _empty_value,
    as_alleeg_list,
    build_python_call,
    ensure_study,
    sync_datasetinfo,
)
from eegprep.functions.studyfunc.std_substudy import std_substudy


def std_rmdat(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    *args: Any,
    datinds: Any = None,
    chanrange: Any = (0, np.inf),
    pntsrange: Any = (0, np.inf),
    sraterange: Any = (0, np.inf),
    trialrange: Any = (1, np.inf),
    checkeventtype: Any = None,
    numeventrange: Any = 1,
    subjectind: Any = None,
    rmvarvalues: Any = None,
    keepvarvalues: Any = None,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Remove datasets from STUDY/ALLEEG and return removed 1-based indices."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    datinds = options.pop("datinds", datinds)
    chanrange = options.pop("chanrange", chanrange)
    pntsrange = options.pop("pntsrange", pntsrange)
    sraterange = options.pop("sraterange", sraterange)
    trialrange = options.pop("trialrange", trialrange)
    checkeventtype = options.pop("checkeventtype", checkeventtype)
    numeventrange = options.pop("numeventrange", numeventrange)
    subjectind = options.pop("subjectind", subjectind)
    rmvarvalues = options.pop("rmvarvalues", rmvarvalues)
    keepvarvalues = options.pop("keepvarvalues", keepvarvalues)
    if options:
        raise ValueError(f"Unknown std_rmdat option(s): {', '.join(sorted(options))}")
    datasets = as_alleeg_list(ALLEEG)
    study = sync_datasetinfo(ensure_study(STUDY), datasets)
    infos = [info for info in study.get("datasetinfo") or [] if isinstance(info, dict)]
    total = max(len(datasets), len(infos))
    remove = set(_index_values(datinds, total))
    remove |= _shape_filtered(datasets, pntsrange, sraterange, chanrange, trialrange)
    if not _empty_value(subjectind):
        keep_subjects = _subjects_from_indices(study, subjectind)
        keepvarvalues = ("subject", keep_subjects)
    remove |= _variable_filtered(infos, rmvarvalues, remove_matches=True)
    remove |= _variable_filtered(infos, keepvarvalues, remove_matches=False)
    remove |= _event_filtered(datasets, checkeventtype, numeventrange)
    removed = sorted(remove)
    command = _history_command(
        datinds,
        chanrange,
        pntsrange,
        sraterange,
        trialrange,
        checkeventtype,
        numeventrange,
        subjectind,
        rmvarvalues,
        keepvarvalues,
    )
    if not removed:
        return (study, datasets, removed, command) if return_com else (study, datasets, removed)
    keep = [index for index in range(1, total + 1) if index not in remove]
    output_study, output_alleeg, _sub_command = std_substudy(study, datasets, dataset=keep, rmdat="on", return_com=True)
    return (output_study, output_alleeg, removed, command) if return_com else (output_study, output_alleeg, removed)


def _shape_filtered(
    datasets: list[dict[str, Any]], pntsrange: Any, sraterange: Any, chanrange: Any, trialrange: Any
) -> set[int]:
    pnts = _range_pair(pntsrange)
    srate = _range_pair(sraterange)
    chans = _range_pair(chanrange)
    trials = _range_pair(trialrange)
    remove = set()
    for index, eeg in enumerate(datasets, start=1):
        if not _in_range(eeg.get("pnts", 0), pnts):
            remove.add(index)
        if not _in_range(eeg.get("srate", 0), srate):
            remove.add(index)
        if not _in_range(eeg.get("nbchan", 0), chans):
            remove.add(index)
        if not _in_range(eeg.get("trials", 1), trials):
            remove.add(index)
    return remove


def _variable_filtered(infos: list[dict[str, Any]], spec: Any, *, remove_matches: bool) -> set[int]:
    if _empty_value(spec):
        return set()
    label, values = _variable_spec(spec)
    matched = {index for index, info in enumerate(infos, start=1) if _value_matches(info.get(label), values)}
    if remove_matches:
        return matched
    return set(range(1, len(infos) + 1)) - matched


def _value_matches(value: Any, selected: Any) -> bool:
    if isinstance(selected, np.ndarray):
        selected = selected.tolist()
    if isinstance(selected, str):
        return str(value) == selected
    if isinstance(selected, (list, tuple, set)):
        if len(selected) == 2 and all(_is_number(item) for item in selected):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return False
            return float(selected[0]) <= numeric <= float(selected[1])
        return any(_value_matches(value, item) for item in selected)
    return value == selected


def _event_filtered(datasets: list[dict[str, Any]], eventtypes: Any, numeventrange: Any) -> set[int]:
    values = _value_list(eventtypes)
    if not values:
        return set()
    bounds = _count_range(numeventrange)
    remove = set()
    for index, eeg in enumerate(datasets, start=1):
        event_rows = _event_rows(eeg.get("event"))
        event_values = [event.get("type") for event in event_rows]
        for event_type in values:
            count = sum(1 for value in event_values if value == event_type)
            if count < bounds[0] or count > bounds[1]:
                remove.add(index)
                break
    return remove


def _subjects_from_indices(study: dict[str, Any], indices: Any) -> list[Any]:
    subjects = list(study.get("subject") or [])
    selected = _index_values(indices, len(subjects))
    return [subjects[index - 1] for index in selected]


def _variable_spec(spec: Any) -> tuple[str, Any]:
    if isinstance(spec, np.ndarray):
        spec = spec.tolist()
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        raise ValueError("variable value filters must be [name, values]")
    return str(spec[0]), spec[1]


def _range_pair(value: Any) -> tuple[float, float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("range options must contain [min max]")
        return float(value[0]), float(value[1])
    return float(value), np.inf


def _count_range(value: Any) -> tuple[int, float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("numeventrange must contain [min max]")
        return int(value[0]), float(value[1])
    return int(value), np.inf


def _index_values(value: Any, total: int) -> list[int]:
    if _empty_value(value):
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    indices = [int(item) for item in value]
    invalid = [index for index in indices if index < 1 or index > total]
    if invalid:
        raise ValueError(f"dataset indices out of range: {invalid}")
    return indices


def _value_list(value: Any) -> list[Any]:
    if _empty_value(value):
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _event_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    return [event for event in value if isinstance(event, dict)]


def _in_range(value: Any, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= float(value) <= bounds[1]


def _is_number(value: Any) -> bool:
    return isinstance(value, (np.integer, np.floating, int, float)) and not isinstance(value, bool)


def _history_command(
    datinds: Any,
    chanrange: Any,
    pntsrange: Any,
    sraterange: Any,
    trialrange: Any,
    checkeventtype: Any,
    numeventrange: Any,
    subjectind: Any,
    rmvarvalues: Any,
    keepvarvalues: Any,
) -> str:
    return build_python_call(
        ("STUDY", "ALLEEG", "RMDATS"),
        "std_rmdat",
        "STUDY",
        "ALLEEG",
        datinds=datinds,
        chanrange=chanrange,
        pntsrange=pntsrange,
        sraterange=sraterange,
        trialrange=trialrange,
        checkeventtype=checkeventtype,
        numeventrange=numeventrange,
        subjectind=subjectind,
        rmvarvalues=rmvarvalues,
        keepvarvalues=keepvarvalues,
    )


__all__ = ["std_rmdat"]
