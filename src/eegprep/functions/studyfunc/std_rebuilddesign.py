"""Rebuild STUDY design values from current dataset metadata."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.studyfunc._study_utils import (
    as_alleeg_list,
    build_python_call,
    ensure_study,
    sync_datasetinfo,
    variable_values,
)
from eegprep.functions.studyfunc.std_addvarlevel import std_addvarlevel


def std_rebuilddesign(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None = None,
    designind: int | None = None,
    *,
    return_com: bool = False,
) -> Any:
    """Refresh STUDY design variables after dataset metadata changes."""
    datasets = as_alleeg_list(ALLEEG)
    study = sync_datasetinfo(deepcopy(ensure_study(STUDY)), datasets)
    designs = list(study.get("design") or [])
    indices = range(1, len(designs) + 1) if designind is None else [int(designind)]
    for index in indices:
        if index < 1 or index > len(designs):
            raise ValueError(f"designind must be 1-based and within 1..{len(designs)}")
        design = designs[index - 1]
        for variable in design.get("variable") or []:
            if not isinstance(variable, dict):
                continue
            values = variable_values(study, str(variable.get("label") or ""))
            if str(variable.get("vartype") or "categorical").lower() == "categorical":
                variable["value"] = values
                variable["levelindex"] = list(range(1, len(values) + 1))
        designs[index - 1] = design
    study["design"] = designs
    study = std_addvarlevel(study, designind)
    study["saved"] = "no"
    kwargs = {"designind": designind} if designind is not None else {}
    command = build_python_call(("STUDY",), "std_rebuilddesign", "STUDY", "ALLEEG", **kwargs)
    return (study, command) if return_com else study


__all__ = ["std_rebuilddesign"]
