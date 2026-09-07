"""Select a STUDY independent variable from available factor metadata."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc.plot_utils import python_literal
from eegprep.functions.studyfunc._study_utils import (
    build_python_call,
    ensure_study,
    parse_design_values,
    variable_values,
)


def pop_addindepvar(
    varlist: dict[str, Any] | list[str],
    fig: Any | None = None,
    var: str | None = None,
    values: Any = None,
    *,
    vartype: str | None = None,
    return_com: bool = False,
) -> Any:
    """Return ``(variable, values, categorical_flag)`` for STUDY designs.

    EEGPrep implements the workflow-supporting non-GUI behavior of EEGLAB's
    helper. Callback strings and MATLAB figure mutation are intentionally not
    emulated.
    """
    if isinstance(varlist, str):
        raise NotImplementedError("pop_addindepvar MATLAB GUI callbacks are not available in EEGPrep")
    if fig is not None:
        raise NotImplementedError("pop_addindepvar GUI figure callbacks are not available in EEGPrep")
    factors, factor_values, numerical = _factor_lists(varlist)
    if not factors:
        raise ValueError("No STUDY independent variables are available")
    label = str(var or factors[0])
    if label not in factors:
        raise ValueError(f"Unknown STUDY independent variable: {label}")
    index = factors.index(label)
    selected_vartype = str(vartype or ("continuous" if numerical[index] else "categorical")).lower()
    if selected_vartype not in {"categorical", "continuous"}:
        raise ValueError("vartype must be 'categorical' or 'continuous'")
    if selected_vartype == "continuous":
        command = _history_command(varlist, var=var, values=values, vartype=vartype)
        return (label, [], 0, command) if return_com else (label, [], 0)
    selected_values = parse_design_values(values)
    if not selected_values:
        selected_values = list(factor_values[index])
    missing = [value for value in selected_values if value not in factor_values[index]]
    if missing:
        raise ValueError(f"Unknown value(s) for STUDY variable {label}: {missing}")
    command = _history_command(varlist, var=var, values=values, vartype=vartype)
    return (label, selected_values, 1, command) if return_com else (label, selected_values, 1)


def _history_command(varlist: dict[str, Any] | list[str], **kwargs: Any) -> str:
    source = "STUDY" if isinstance(varlist, dict) and "datasetinfo" in varlist else python_literal(varlist)
    return build_python_call(("variable", "values", "categorical"), "pop_addindepvar", source, **kwargs)


def _factor_lists(varlist: dict[str, Any] | list[str]) -> tuple[list[str], list[list[Any]], list[bool]]:
    if isinstance(varlist, list):
        return [str(item) for item in varlist], [[] for _item in varlist], [False for _item in varlist]
    if "factors" in varlist:
        factors = [str(item) for item in varlist.get("factors") or []]
        values = [list(item) if isinstance(item, (list, tuple)) else [item] for item in varlist.get("factorvals") or []]
        while len(values) < len(factors):
            values.append([])
        numerical = [bool(item) for item in varlist.get("numerical") or []]
        while len(numerical) < len(factors):
            numerical.append(False)
        return factors, values, numerical
    study = ensure_study(varlist)
    factors = []
    values = []
    numerical = []
    for label in _study_factor_names(study):
        factor_values = variable_values(study, label)
        factors.append(label)
        values.append(factor_values)
        numerical.append(bool(factor_values) and all(isinstance(value, (int, float)) for value in factor_values))
    return factors, values, numerical


def _study_factor_names(study: dict[str, Any]) -> list[str]:
    names = []
    for info in study.get("datasetinfo") or []:
        if not isinstance(info, dict):
            continue
        for key, value in info.items():
            if key not in {"index", "setname", "filename", "filepath", "comps"} and value not in (None, "", []):
                if key not in names:
                    names.append(key)
    return names


__all__ = ["pop_addindepvar"]
