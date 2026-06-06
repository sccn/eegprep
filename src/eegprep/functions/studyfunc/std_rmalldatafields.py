"""Remove cached data fields from STUDY channel or cluster entries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.studyfunc._study_utils import STUDY_DATA_FIELDS, ensure_study


def std_rmalldatafields(STUDY: dict[str, Any], chanorcomp: str = "both") -> dict[str, Any]:
    """Return STUDY without cached measure/data fields for the requested target."""
    study = ensure_study(STUDY)
    target = str(chanorcomp or "both").lower()
    if target not in {"chan", "channels", "data", "clust", "cluster", "components", "both"}:
        raise ValueError("chanorcomp must be 'chan', 'clust', 'data', or 'both'")
    output = deepcopy(study)
    fields = set(STUDY_DATA_FIELDS)
    if target in {"chan", "channels", "data", "both"}:
        _remove_fields(output.get("changrp"), fields)
    if target in {"clust", "cluster", "components", "both"}:
        _remove_fields(output.get("cluster"), fields)
    return output


def _remove_fields(value: Any, fields: set[str]) -> None:
    if isinstance(value, dict):
        for field in fields:
            value.pop(field, None)
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                for field in fields:
                    item.pop(field, None)


__all__ = ["std_rmalldatafields"]
