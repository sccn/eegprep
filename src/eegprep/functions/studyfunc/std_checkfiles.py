"""Check STUDY dataset files and cached measure consistency."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.studyfunc._study_utils import as_alleeg_list, check_datasetinfo, ensure_study
from eegprep.functions.studyfunc.std_uniformfiles import std_uniformfiles
from eegprep.functions.studyfunc.std_uniformsetinds import std_uniformsetinds


def std_checkfiles(
    STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None = None, *, return_report: bool = False
) -> Any:
    """Check loaded STUDY data and cached measures for standalone EEGPrep."""
    study = ensure_study(STUDY)
    datasets = as_alleeg_list(ALLEEG)
    report = check_datasetinfo(study, datasets)
    report["uniformfiles"] = std_uniformfiles(study, datasets)
    report["uniformsetinds"] = std_uniformsetinds(study)
    report["measure_cache"] = _measure_cache_report(study, len(datasets))
    if report["uniformfiles"] == 0:
        report["problems"].append("loaded datasets do not share one channel-label distribution")
    if report["uniformfiles"] == -1:
        report["problems"].append("one or more STUDY dataset files are missing")
    if report["uniformsetinds"] == 0:
        report["problems"].append("STUDY.changrp set indices are not uniform")
    for problem in report["measure_cache"]["problems"]:
        report["problems"].append(problem)
    ok = not report["problems"]
    return (int(ok), report) if return_report else int(ok)


def _measure_cache_report(study: dict[str, Any], dataset_count: int) -> dict[str, Any]:
    problems = []
    checked = []
    for group_name in ("changrp", "cluster"):
        for index, group in enumerate(study.get(group_name) or [], start=1):
            if not isinstance(group, dict):
                continue
            for field in ("erpdata", "specdata", "erspdata", "itcdata", "topo", "pacdata"):
                if field not in group:
                    continue
                array = np.asarray(group[field], dtype=float)
                checked.append({"group": group_name, "index": index, "field": field, "shape": list(array.shape)})
                if array.size == 0:
                    problems.append(f"{group_name}[{index}].{field} is empty")
                if dataset_count and array.ndim and group_name == "changrp" and array.shape[0] != dataset_count:
                    problems.append(f"{group_name}[{index}].{field} dataset axis does not match ALLEEG")
    return {"checked": checked, "problems": problems}


__all__ = ["std_checkfiles"]
