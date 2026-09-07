"""Check whether loaded STUDY datasets have uniform channel files."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc.plot_utils import channel_labels
from eegprep.functions.studyfunc._study_utils import as_alleeg_list, check_datasetinfo, ensure_study


def std_uniformfiles(STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None) -> int:
    """Return ``1`` for uniform channels, ``0`` for mismatch, ``-1`` for missing files."""
    study = ensure_study(STUDY)
    datasets = as_alleeg_list(ALLEEG)
    consistency = check_datasetinfo(study, datasets)
    if consistency["missing_files"]:
        return -1
    if not datasets:
        return 1
    labels = [channel_labels(eeg) for eeg in datasets]
    return 1 if len({tuple(item) for item in labels}) <= 1 else 0


__all__ = ["std_uniformfiles"]
