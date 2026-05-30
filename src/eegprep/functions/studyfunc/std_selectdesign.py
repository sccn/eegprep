"""Select an existing STUDY design."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_utils import clear_study_data_fields, ensure_study


def std_selectdesign(STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None, designind: int) -> dict[str, Any]:
    """Select a 1-based STUDY design index."""
    study = ensure_study(STUDY)
    designs = study.get("design") or []
    index = int(designind)
    if index < 1 or index > len(designs) or not designs[index - 1].get("name"):
        raise ValueError("Cannot select an empty STUDY.design")
    study["currentdesign"] = index
    study = clear_study_data_fields(study)
    return study


__all__ = ["std_selectdesign"]
