"""Select an existing STUDY design."""

from __future__ import annotations

from typing import Any

from eegprep.functions.miscfunc.matlab_inputs import struct_records
from eegprep.functions.studyfunc._study_utils import clear_study_data_fields, ensure_study


def std_selectdesign(STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None, designind: int) -> dict[str, Any]:
    """Select a 1-based STUDY design index."""
    study = ensure_study(STUDY)
    designs = _design_list(study.get("design"))
    index = int(designind)
    if index < 1 or index > len(designs) or not designs[index - 1].get("name"):
        raise ValueError("Cannot select an empty STUDY.design")
    study["currentdesign"] = index
    study = clear_study_data_fields(study)
    return study


def _design_list(design: Any) -> list[dict[str, Any]]:
    records = struct_records(design)
    if records is not None:
        return records
    return []


__all__ = ["std_selectdesign"]
