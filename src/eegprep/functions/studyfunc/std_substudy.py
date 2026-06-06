"""Create a STUDY subset by dataset, subject, condition, or group."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._cluster_utils import sets_array
from eegprep.functions.studyfunc._study_utils import (
    _empty_value,
    as_alleeg_list,
    build_python_call,
    ensure_study,
    sync_datasetinfo,
)
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.studyfunc.std_rmalldatafields import std_rmalldatafields


def std_substudy(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    *args: Any,
    dataset: Any = None,
    subject: Any = None,
    condition: Any = None,
    group: Any = None,
    rmdat: str | bool = "on",
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Return a STUDY/ALLEEG subset using EEGLAB-facing 1-based selectors."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    dataset = options.pop("dataset", dataset)
    subject = options.pop("subject", subject)
    condition = options.pop("condition", condition)
    group = options.pop("group", group)
    rmdat = options.pop("rmdat", rmdat)
    if options:
        raise ValueError(f"Unknown std_substudy option(s): {', '.join(sorted(options))}")
    datasets = as_alleeg_list(ALLEEG)
    study = sync_datasetinfo(ensure_study(STUDY), datasets)
    infos = [info for info in study.get("datasetinfo") or [] if isinstance(info, dict)]
    keep = _kept_dataset_indices(infos, len(datasets), dataset, subject, condition, group)
    if not keep:
        raise ValueError("std_substudy selection removed every dataset")
    remove_data = is_on(rmdat)
    mapping = {
        old_index: (new_index if remove_data else old_index) for new_index, old_index in enumerate(keep, start=1)
    }

    output = deepcopy(study)
    if remove_data:
        output["datasetinfo"] = [deepcopy(infos[index - 1]) for index in keep]
        for index, info in enumerate(output["datasetinfo"], start=1):
            info["index"] = index
        output_alleeg = [deepcopy(datasets[index - 1]) for index in keep if index <= len(datasets)]
    else:
        output["datasetinfo"] = deepcopy(infos)
        output_alleeg = deepcopy(datasets)
    _remap_dataset_references(output, mapping)
    output = std_rmalldatafields(output, "both")
    output["cache"] = []
    output["saved"] = "no"
    checked, checked_alleeg = std_checkset(output, output_alleeg)
    command = _history_command(dataset, subject, condition, group, rmdat)
    return (checked, checked_alleeg, command) if return_com else (checked, checked_alleeg)


def _kept_dataset_indices(
    infos: list[dict[str, Any]],
    dataset_count: int,
    dataset: Any,
    subject: Any,
    condition: Any,
    group: Any,
) -> list[int]:
    total = max(len(infos), dataset_count)
    keep = set(range(1, total + 1))
    if not _empty_value(dataset):
        keep &= set(_index_values(dataset, total))
    for label, selector in (("subject", subject), ("condition", condition), ("group", group)):
        values = {str(value) for value in _value_list(selector)}
        if values:
            keep &= {index for index, info in enumerate(infos, start=1) if str(info.get(label) or "") in values}
    return sorted(keep)


def _index_values(value: Any, total: int) -> list[int]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
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


def _remap_dataset_references(study: dict[str, Any], mapping: dict[int, int]) -> None:
    for group in study.get("changrp") or []:
        if not isinstance(group, dict):
            continue
        group["sets"] = _remap_vector(group.get("sets"), mapping)
        measureinfo = group.get("measureinfo")
        if isinstance(measureinfo, dict):
            measureinfo["datasets"] = _remap_vector(measureinfo.get("datasets"), mapping)
    for cluster in study.get("cluster") or []:
        if not isinstance(cluster, dict):
            continue
        sets = sets_array(cluster.get("sets"))
        comps = np.asarray(cluster.get("comps") or [], dtype=int).ravel()
        if sets.size and comps.size:
            remapped = np.vectorize(lambda value: 0 if np.isnan(value) else mapping.get(int(value), 0), otypes=[int])(
                sets
            )
            keep_columns = np.any(remapped != 0, axis=0)
            cluster["sets"] = remapped[:, keep_columns].tolist()
            cluster["comps"] = comps[keep_columns].astype(int).tolist()
        else:
            cluster["sets"] = []
            cluster["comps"] = []
        measureinfo = cluster.get("measureinfo")
        if isinstance(measureinfo, dict):
            measureinfo["datasets"] = _remap_vector(measureinfo.get("datasets"), mapping)


def _remap_vector(value: Any, mapping: dict[int, int]) -> list[int]:
    if _empty_value(value):
        return []
    array = np.asarray(value, dtype=float).ravel()
    output = []
    for item in array:
        if np.isnan(item):
            continue
        mapped = mapping.get(int(item))
        if mapped and mapped not in output:
            output.append(mapped)
    return output


def _history_command(dataset: Any, subject: Any, condition: Any, group: Any, rmdat: Any) -> str:
    kwargs = {
        "dataset": dataset,
        "subject": subject,
        "condition": condition,
        "group": group,
        "rmdat": "on" if is_on(rmdat) else "off",
    }
    return build_python_call(("STUDY", "ALLEEG"), "std_substudy", "STUDY", "ALLEEG", **kwargs)


__all__ = ["std_substudy"]
