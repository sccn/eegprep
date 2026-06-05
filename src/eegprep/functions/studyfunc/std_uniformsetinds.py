"""Check whether STUDY channel groups use uniform dataset indices."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.studyfunc._study_utils import ensure_study


def std_uniformsetinds(STUDY: dict[str, Any]) -> int:
    """Return ``1`` when STUDY channel group set indices are uniform."""
    study = ensure_study(STUDY)
    groups = [group for group in study.get("changrp") or [] if isinstance(group, dict)]
    if len(groups) < 2:
        return 1
    reference = _set_signature(groups[0])
    for group in groups[1:]:
        if _set_signature(group) != reference:
            return 0
    return 1


def _set_signature(group: dict[str, Any]) -> tuple[int, ...]:
    values = group.get("sets")
    if values is None:
        values = group.get("setinds")
    if values is None:
        return ()
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return ()
    return tuple(int(item) for item in array.ravel().tolist())


__all__ = ["std_uniformsetinds"]
