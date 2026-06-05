"""Import a subject-level variable into a STUDY."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from eegprep.functions.studyfunc._study_utils import build_python_call, ensure_study
from eegprep.functions.studyfunc.std_addvarlevel import std_addvarlevel


def pop_importgroupvar(
    STUDY: dict[str, Any],
    designind: int = 1,
    *,
    variable: str,
    values: Any = None,
    filepath: str | Path | None = None,
    vartype: str = "categorical",
    return_com: bool = False,
) -> Any:
    """Attach one imported variable value per STUDY subject.

    Values may be a ``{subject: value}`` mapping, a sequence in design subject
    order, or a text file containing one value per subject.
    """
    study = deepcopy(ensure_study(STUDY))
    label = str(variable or "").strip()
    if not label:
        raise ValueError("variable must be provided")
    subjects = _design_subjects(study, int(designind))
    if not subjects:
        subjects = [str(info.get("subject") or "") for info in study.get("datasetinfo") or []]
    imported = _imported_values(subjects, values, filepath)
    for info in study.get("datasetinfo") or []:
        if not isinstance(info, dict):
            continue
        subject = str(info.get("subject") or "")
        if subject in imported:
            info[label] = imported[subject]
    study = _update_design(study, int(designind), label, list(imported.values()), vartype)
    study = std_addvarlevel(study, int(designind))
    study["saved"] = "no"
    command = build_python_call(
        ("STUDY",),
        "pop_importgroupvar",
        "STUDY",
        str(int(designind)),
        variable=label,
        values=imported,
        vartype=vartype,
    )
    return (study, command) if return_com else study


def _design_subjects(study: dict[str, Any], designind: int) -> list[str]:
    designs = study.get("design") or []
    if designind < 1 or designind > len(designs):
        return []
    cases = designs[designind - 1].get("cases") or {}
    return [str(value) for value in cases.get("value") or []]


def _imported_values(subjects: list[str], values: Any, filepath: str | Path | None) -> dict[str, Any]:
    if filepath is not None:
        raw_values = [line.strip() for line in Path(filepath).read_text(encoding="utf-8").splitlines() if line.strip()]
    elif isinstance(values, dict):
        missing = [subject for subject in subjects if subject not in values]
        if missing:
            raise ValueError(f"Missing imported values for subject(s): {', '.join(missing)}")
        return {subject: values[subject] for subject in subjects}
    else:
        raw_values = list(values or [])
    if len(raw_values) != len(subjects):
        raise ValueError("Number of imported values must match the number of STUDY subjects")
    return dict(zip(subjects, raw_values))


def _update_design(
    study: dict[str, Any], designind: int, label: str, values: list[Any], vartype: str
) -> dict[str, Any]:
    designs = list(study.get("design") or [])
    if designind < 1:
        raise ValueError("designind must be a 1-based positive integer")
    while len(designs) < designind:
        designs.append({"name": f"STUDY.design {len(designs) + 1}", "variable": [], "cases": {}})
    design = designs[designind - 1]
    variables = [variable for variable in design.get("variable") or [] if isinstance(variable, dict)]
    variables = [variable for variable in variables if variable.get("label") != label]
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    variables.append(
        {
            "label": label,
            "value": unique_values if vartype == "categorical" else [],
            "vartype": vartype,
            "pairing": "off",
            "levelindex": list(range(1, len(unique_values) + 1)),
        }
    )
    design["variable"] = variables
    designs[designind - 1] = design
    study["design"] = designs
    return study


__all__ = ["pop_importgroupvar"]
