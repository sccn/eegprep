"""Check STUDY metadata consistency."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_utils import as_alleeg_list, ensure_study, store_consistency, sync_datasetinfo
from eegprep.functions.studyfunc.std_makedesign import std_makedesign


def std_checkset(
    STUDY: dict[str, Any] | None,
    ALLEEG: list[dict[str, Any]] | None,
    *,
    return_com: bool = False,
) -> Any:
    """Normalize a STUDY dict and refresh dataset/design consistency metadata."""
    datasets = as_alleeg_list(ALLEEG)
    study = sync_datasetinfo(ensure_study(STUDY), datasets)
    if not study.get("design") and study.get("datasetinfo"):
        study, _command = std_makedesign(study, datasets, 1, return_com=True)
    elif study.get("design") and not study.get("currentdesign"):
        study["currentdesign"] = 1
    study = store_consistency(study, datasets)
    command = "STUDY, ALLEEG = std_checkset(STUDY, ALLEEG)" if return_com else ""
    return (study, datasets, command) if return_com else (study, datasets)


def std_checkdatasetinfo(STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Return STUDY.datasetinfo consistency checks for loaded datasets."""
    datasets = as_alleeg_list(ALLEEG)
    study = sync_datasetinfo(ensure_study(STUDY), datasets)
    if "dataset_consistency" not in study["etc"]["eegprep"]:
        study = store_consistency(study, datasets)
    return study["etc"]["eegprep"]["dataset_consistency"]


__all__ = ["std_checkdatasetinfo", "std_checkset"]
