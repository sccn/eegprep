"""Create or edit STUDY design metadata."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.studyfunc._study_utils import (
    as_alleeg_list,
    available_variables,
    build_python_call,
    ensure_study,
    parse_design_values,
    store_consistency,
    sync_datasetinfo,
    variable_values,
)
from eegprep.functions.studyfunc.std_selectdesign import std_selectdesign


def std_makedesign(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    designind: int = 1,
    *args: Any,
    name: str | None = None,
    variable1: str | None = None,
    values1: Any = None,
    vartype1: str = "categorical",
    variable2: str | None = None,
    values2: Any = None,
    vartype2: str = "categorical",
    variable3: str | None = None,
    values3: Any = None,
    vartype3: str = "categorical",
    variable4: str | None = None,
    values4: Any = None,
    vartype4: str = "categorical",
    subjselect: Any = None,
    filepath: str = "",
    datselect: Any = None,
    delfiles: str = "off",
    defaultdesign: str = "on",
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Create or replace a 1-based STUDY design.

    This Phase 5a implementation stores design metadata and factor selections.
    Measure-file deletion/precompute side effects from EEGLAB are intentionally
    outside this phase.
    """
    datasets = as_alleeg_list(ALLEEG)
    study = sync_datasetinfo(ensure_study(STUDY), datasets)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    name = options.pop("name", name)
    variable1 = options.pop("variable1", variable1)
    values1 = options.pop("values1", values1)
    vartype1 = options.pop("vartype1", vartype1)
    variable2 = options.pop("variable2", variable2)
    values2 = options.pop("values2", values2)
    vartype2 = options.pop("vartype2", vartype2)
    variable3 = options.pop("variable3", variable3)
    values3 = options.pop("values3", values3)
    vartype3 = options.pop("vartype3", vartype3)
    variable4 = options.pop("variable4", variable4)
    values4 = options.pop("values4", values4)
    vartype4 = options.pop("vartype4", vartype4)
    subjselect = options.pop("subjselect", subjselect)
    filepath = str(options.pop("filepath", filepath) or "")
    datselect = options.pop("datselect", options.pop("dataselect", datselect))
    delfiles = str(options.pop("delfiles", delfiles) or "off")
    defaultdesign = str(options.pop("defaultdesign", defaultdesign) or "on")
    if options:
        raise ValueError(f"Unknown std_makedesign option(s): {', '.join(sorted(options))}")
    if delfiles not in {"on", "off", "limited"}:
        raise ValueError("delfiles must be 'on', 'off', or 'limited'")

    design_index = int(designind)
    if design_index < 1:
        raise ValueError("designind must be a 1-based positive integer")

    variables = _design_variables(
        study,
        (
            (variable1, values1, vartype1),
            (variable2, values2, vartype2),
            (variable3, values3, vartype3),
            (variable4, values4, vartype4),
        ),
        defaultdesign=defaultdesign,
    )
    design = {
        "name": name or f"STUDY.design {design_index}",
        "filepath": filepath,
        "variable": variables,
        "cases": {"label": "subject", "value": _subject_values(study, subjselect)},
        "include": [] if datselect is None else datselect,
    }
    designs = list(study.get("design") or [])
    while len(designs) < design_index:
        designs.append(_empty_design(len(designs) + 1))
    designs[design_index - 1] = design
    study["design"] = designs
    study = std_selectdesign(study, datasets, design_index)
    study["cache"] = []
    study["saved"] = "no"
    study = store_consistency(study, datasets)
    command = _history_command(
        design_index,
        design,
        delfiles=delfiles,
        defaultdesign=defaultdesign,
        filepath=filepath,
        datselect=datselect,
    )
    return (study, command) if return_com else study


def _design_variables(
    study: dict[str, Any],
    specs: tuple[tuple[str | None, Any, str], ...],
    *,
    defaultdesign: str,
) -> list[dict[str, Any]]:
    labels = available_variables(study)
    if all(_empty_label(label) for label, _values, _vartype in specs) and defaultdesign != "forceoff":
        specs = _default_specs(study, labels)
    variables = []
    for label, values, vartype in specs:
        if _empty_label(label):
            continue
        label = str(label)
        if label not in labels:
            raise ValueError(f"Unknown STUDY design variable: {label}")
        parsed_values = parse_design_values(values)
        if not parsed_values:
            parsed_values = variable_values(study, label)
        variable = {
            "label": label,
            "value": parsed_values,
            "vartype": _vartype(vartype),
            "pairing": "off" if label == "group" else "on",
            "level": list(range(1, len(parsed_values) + 1)),
        }
        variables.append(variable)
    return variables


def _default_specs(study: dict[str, Any], labels: list[str]) -> tuple[tuple[str | None, Any, str], ...]:
    varying_labels = [label for label in labels if len(variable_values(study, label)) > 1]
    first = "condition" if "condition" in varying_labels else (varying_labels[0] if varying_labels else None)
    second = "group" if "group" in varying_labels and first != "group" else None
    return ((first, None, "categorical"), (second, None, "categorical"), (None, None, ""), (None, None, ""))


def _subject_values(study: dict[str, Any], subjselect: Any) -> list[Any]:
    if subjselect is None or subjselect == []:
        return list(study.get("subject") or [])
    values = parse_design_values(subjselect)
    unknown = sorted(set(values) - set(study.get("subject") or []))
    if unknown:
        raise ValueError(f"Unknown STUDY subject(s): {', '.join(str(item) for item in unknown)}")
    return values


def _empty_design(index: int) -> dict[str, Any]:
    return {
        "name": f"STUDY.design {index}",
        "filepath": "",
        "variable": [],
        "cases": {"label": "subject", "value": []},
        "include": [],
    }


def _history_command(
    design_index: int,
    design: dict[str, Any],
    *,
    delfiles: str,
    defaultdesign: str,
    filepath: str,
    datselect: Any,
) -> str:
    kwargs: dict[str, Any] = {"name": design["name"], "delfiles": delfiles, "defaultdesign": defaultdesign}
    for index, variable in enumerate(design.get("variable") or [], start=1):
        kwargs[f"variable{index}"] = variable["label"]
        kwargs[f"values{index}"] = variable.get("value", [])
        kwargs[f"vartype{index}"] = variable.get("vartype", "categorical")
    cases = design.get("cases", {}).get("value", [])
    if cases:
        kwargs["subjselect"] = cases
    if filepath:
        kwargs["filepath"] = filepath
    if datselect:
        kwargs["datselect"] = datselect
    return build_python_call(("STUDY",), "std_makedesign", "STUDY", "ALLEEG", str(design_index), **kwargs)


def _empty_label(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "[]"}


def _vartype(value: Any) -> str:
    text = str(value or "categorical").lower()
    if text not in {"categorical", "continuous"}:
        raise ValueError("vartype must be 'categorical' or 'continuous'")
    return text


__all__ = ["std_makedesign"]
