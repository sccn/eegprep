"""Find dataset-level variables available for STUDY designs."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_utils import RESERVED_VARIABLE_FIELDS, _empty_value, ensure_study


def std_findgroupvars(STUDY: dict[str, Any]) -> list[str]:
    """Return non-trial STUDY.datasetinfo variables with usable values."""
    study = ensure_study(STUDY)
    names: list[str] = []
    for info in study.get("datasetinfo") or []:
        if not isinstance(info, dict):
            continue
        for key, value in info.items():
            if key in RESERVED_VARIABLE_FIELDS or key == "subject" or _empty_value(value):
                continue
            if key not in names:
                names.append(key)
    preferred = [field for field in ("condition", "group", "session", "run") if field in names]
    return preferred + sorted(name for name in names if name not in preferred)


__all__ = ["std_findgroupvars"]
