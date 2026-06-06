"""Check whether a STUDY design fits basic plotting assumptions."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_utils import ensure_study


def std_checkdesign(STUDY: dict[str, Any], designind: int | None = None) -> int:
    """Return 1 when a design has no continuous variables or multi-valued extras."""
    study = ensure_study(STUDY)
    index = int(designind if designind is not None else study.get("currentdesign") or 1)
    designs = study.get("design") or []
    if index < 1 or index > len(designs):
        raise ValueError(f"designind must be 1-based and within 1..{len(designs)}")
    variables = [variable for variable in designs[index - 1].get("variable") or [] if isinstance(variable, dict)]
    if any(str(variable.get("vartype") or "categorical").lower() == "continuous" for variable in variables):
        return 0
    for variable in variables[2:]:
        values = variable.get("value") or []
        if len(values) > 1:
            return 0
    return 1


__all__ = ["std_checkdesign"]
