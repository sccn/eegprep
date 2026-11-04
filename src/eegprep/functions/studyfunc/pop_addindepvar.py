"""Select a STUDY independent variable from available factor metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc.plot_utils import python_literal
from eegprep.functions.studyfunc._study_utils import (
    build_python_call,
    ensure_study,
    parse_design_values,
    variable_values,
)
from eegprep.functions.studyfunc.std_indvarmatch import std_indvarmatch


_CATEGORICAL_POPUP = "This is a categorical var.|This is a continuous var."
_VALUE_SEPARATOR = " && "


def pop_addindepvar(
    varlist: dict[str, Any] | list[str],
    fig: Any | None = None,
    var: str | None = None,
    values: Any = None,
    *,
    vartype: str | None = None,
    gui: bool = False,
    renderer: Any | None = None,
    return_com: bool = False,
) -> Any:
    """Return ``(variable, values, categorical_flag)`` for STUDY designs.

    Pass ``gui=True`` to open the Add variable dialog used by
    ``pop_studydesign``. Command-line selection stays available when ``var``
    is set. MATLAB callback strings and figure handles are not evaluated.
    """
    if isinstance(varlist, str):
        raise NotImplementedError("pop_addindepvar MATLAB GUI callbacks are not available in EEGPrep")
    if fig is not None:
        raise NotImplementedError("pop_addindepvar GUI figure callbacks are not available in EEGPrep")
    if gui:
        return _pop_addindepvar_gui(varlist, var=var, values=values, renderer=renderer, return_com=return_com)
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


def pop_addindepvar_dialog_spec(
    varlist: dict[str, Any] | list[str],
    var: str | None = None,
    values: Any = None,
) -> DialogSpec:
    """Return the Add variable dialog used by ``pop_studydesign``."""
    normalized = normalize_addindepvar_varlist(varlist)
    state = {
        "factors": normalized["factors"],
        "factorvals": normalized["factorvals"],
        "numerical": normalized["numerical"],
    }
    factor_index, value_indices, vartype_index = _initial_selection(normalized, var, values)
    values_enabled = vartype_index == 1
    value_labels = _encoded_values(normalized["factorvals"][factor_index - 1])
    controls = (
        ControlSpec("text", "Select independent variable", font_weight="bold"),
        ControlSpec(
            "listbox",
            "|".join(normalized["factors"]),
            tag="lbfact0",
            value=factor_index,
            callback=CallbackSpec("addindepvar_selectfact", state),
        ),
        ControlSpec(
            "popupmenu",
            _CATEGORICAL_POPUP,
            tag="vartype",
            value=vartype_index,
            callback=CallbackSpec("addindepvar_vartype", state),
        ),
        ControlSpec("text", "Select variable values ", tag="text1", enabled=values_enabled),
        ControlSpec(
            "listbox",
            "|".join(value_labels),
            tag="lbval0",
            value=value_indices,
            enabled=values_enabled,
        ),
        ControlSpec(
            "pushbutton",
            "Combine selected values",
            tag="combine1",
            callback=CallbackSpec("addindepvar_combine", state),
            enabled=values_enabled,
        ),
    )
    return DialogSpec(
        title="Add variable",
        controls=controls,
        geometry=((1,), (1,), (1,), (1,), (1,), (1,)),
        geomvert=(1, 3, 1, 1, 3, 1),
        function_name="pop_addindepvar",
        eeglab_source="functions/studyfunc/pop_addindepvar.m",
        help_text=None,
        show_help_button=False,
        size=(420, 480),
        content_margins=(24, 18, 24, 12),
        extra_stylesheet="""
            QDialog#pop_addindepvar QLabel,
            QDialog#pop_addindepvar QPushButton,
            QDialog#pop_addindepvar QComboBox,
            QDialog#pop_addindepvar QListWidget {
                font-size: 13px;
            }
            QDialog#pop_addindepvar QListWidget#lbfact0,
            QDialog#pop_addindepvar QListWidget#lbval0 {
                min-height: 96px;
            }
        """,
        known_differences=(
            "MATLAB callback strings such as pop_addindepvar('selectfact', gcbf) are not evaluated.",
            "The dialog has no Help button because the EEGLAB call does not set a help command.",
        ),
    )


def normalize_addindepvar_varlist(varlist: dict[str, Any] | list[str]) -> dict[str, Any]:
    """Return factors, values, and numeric flags for the Add variable dialog."""
    if isinstance(varlist, dict) and "factors" in varlist:
        factors = [str(item) for item in varlist.get("factors") or []]
        factorvals = []
        for item in varlist.get("factorvals") or []:
            if isinstance(item, (list, tuple)):
                factorvals.append([_plain(value) for value in item])
            elif item in (None, ""):
                factorvals.append([])
            else:
                factorvals.append([_plain(item)])
        while len(factorvals) < len(factors):
            factorvals.append([])
        numerical = [bool(item) for item in varlist.get("numerical") or []]
        while len(numerical) < len(factors):
            numerical.append(_values_are_numeric(factorvals[len(numerical)]))
        normalized = {"factors": factors, "factorvals": factorvals, "numerical": numerical}
        if "preview" in varlist:
            normalized["preview"] = varlist["preview"]
        return normalized
    factors, factorvals, numerical = _factor_lists(varlist)
    return {
        "factors": factors,
        "factorvals": [[_plain(value) for value in values] for values in factorvals],
        "numerical": numerical,
    }


def with_none_factor(varlist: dict[str, Any]) -> dict[str, Any]:
    """Prepend EEGLAB's ``None`` choice to an independent-variable list."""
    normalized = normalize_addindepvar_varlist(varlist)
    if normalized["factors"] and normalized["factors"][0] == "None":
        return normalized
    normalized["factors"] = ["None", *normalized["factors"]]
    normalized["factorvals"] = [[], *normalized["factorvals"]]
    normalized["numerical"] = [True, *normalized["numerical"]]
    return normalized


def selection_from_addindepvar_result(varlist: dict[str, Any], result: Mapping[str, Any]) -> tuple[str, list[Any], int]:
    """Map an Add variable dialog result to ``(label, values, categorical)``."""
    normalized = normalize_addindepvar_varlist(varlist)
    chosen = result.get("lbfact0")
    if isinstance(chosen, dict):
        return str(chosen.get("label") or ""), list(chosen.get("values") or []), int(chosen.get("categorical") or 0)
    factors = normalized["factors"]
    try:
        factor_index = int(chosen or 1) - 1
    except (TypeError, ValueError):
        return "", [], 0
    if factor_index < 0 or factor_index >= len(factors):
        return "", [], 0
    label = factors[factor_index]
    try:
        vartype_index = int(result.get("vartype") or 1)
    except (TypeError, ValueError):
        vartype_index = 1
    if label == "None" or vartype_index == 2:
        return label, [], 0
    selected = result.get("lbval0") or []
    if isinstance(selected, int):
        selected = [selected]
    values = normalized["factorvals"][factor_index]
    picked = [values[int(index) - 1] for index in selected if 1 <= int(index) <= len(values)]
    return label, picked, 1


def read_addindepvar_selection(widgets: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[str, list[Any], int]:
    """Read the current Add variable widgets."""
    factor_index = int(widgets["lbfact0"].currentRow())
    factors = state["factors"]
    if factor_index < 0 or factor_index >= len(factors):
        return "", [], 0
    label = str(factors[factor_index])
    categorical = int(widgets["vartype"].currentIndex()) == 0
    if label == "None" or not categorical:
        return label, [], 0
    value_list = widgets["lbval0"]
    selected = sorted(value_list.row(item) for item in value_list.selectedItems())
    values = state["factorvals"][factor_index]
    return label, [values[row] for row in selected if 0 <= row < len(values)], 1


def apply_addindepvar_selectfact(widgets: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    """Refresh values and the categorical/continuous popup for the selected factor."""
    factor_index = int(widgets["lbfact0"].currentRow())
    if factor_index < 0 or factor_index >= len(state["factors"]):
        return
    values = state["factorvals"][factor_index]
    _fill_value_list(widgets["lbval0"], values, None)
    combo = widgets["vartype"]
    combo.blockSignals(True)
    combo.clear()
    if state["numerical"][factor_index]:
        combo.addItems(["Categorical variable", "Continuous variable"])
        combo.setCurrentIndex(1)
    else:
        combo.addItems(["Categorical variable"])
        combo.setCurrentIndex(0)
    combo.blockSignals(False)
    apply_addindepvar_vartype(widgets, state)


def apply_addindepvar_vartype(widgets: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    """Enable value selection only for a categorical variable."""
    del state
    enabled = int(widgets["vartype"].currentIndex()) == 0
    for tag in ("text1", "lbval0", "combine1"):
        widget = widgets.get(tag)
        if widget is not None:
            widget.setEnabled(enabled)


def apply_addindepvar_combine(widgets: Mapping[str, Any], state: dict[str, Any]) -> str | None:
    """Combine the selected categorical values. Return a warning when that is not possible."""
    factor_index = int(widgets["lbfact0"].currentRow())
    if factor_index < 0 or factor_index >= len(state["factors"]):
        return None
    value_list = widgets["lbval0"]
    selected = sorted(value_list.row(item) for item in value_list.selectedItems())
    if len(selected) < 2:
        return "You need to select several values to combine them"
    values = state["factorvals"][factor_index]
    if not isinstance(values, list):
        return "Cannot combine values from numerical variables"
    chosen = [values[row] for row in selected if 0 <= row < len(values)]
    if not chosen:
        return None
    if all(isinstance(item, str) or isinstance(item, (list, tuple)) for item in chosen):
        merged: list[Any] = []
        for item in chosen:
            if isinstance(item, (list, tuple)):
                merged.extend(item)
            else:
                merged.append(item)
        values.append(_unique_values(merged))
    else:
        numbers: list[Any] = []
        for item in chosen:
            if isinstance(item, (list, tuple)):
                numbers.extend(item)
            else:
                numbers.append(item)
        values.append(_unique_values(numbers))
    _fill_value_list(widgets["lbval0"], values, selected)
    apply_addindepvar_vartype(widgets, state)
    return None


def encode_design_value(value: Any, *, separator: str = " & ") -> str:
    """Format one design value the way EEGLAB list boxes do."""
    if isinstance(value, (list, tuple)):
        return separator.join(encode_design_value(item, separator=separator) for item in value)
    if isinstance(value, str):
        return value
    if _is_number(value):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:g}"
    return str(value)


def _pop_addindepvar_gui(
    varlist: dict[str, Any] | list[str],
    *,
    var: str | None,
    values: Any,
    renderer: Any | None,
    return_com: bool,
) -> Any:
    normalized = normalize_addindepvar_varlist(varlist)
    spec = pop_addindepvar_dialog_spec(normalized, var=var, values=values)
    result = inputgui(spec, renderer=renderer)
    if result is None:
        selection = ("", [], 0)
    else:
        selection = selection_from_addindepvar_result(normalized, result)
    if not return_com:
        return selection
    command = ""
    if selection[0]:
        command = _history_command(
            normalized,
            var=selection[0],
            values=selection[1],
            vartype="categorical" if selection[2] else "continuous",
        )
    return (*selection, command)


def _initial_selection(varlist: dict[str, Any], var: str | None, values: Any) -> tuple[int, list[int], int]:
    factors = varlist["factors"]
    factor_index = factors.index(var) + 1 if var and var in factors else 1
    factor_values = varlist["factorvals"][factor_index - 1]
    numerical = bool(varlist["numerical"][factor_index - 1])
    if values not in (None, "", []):
        requested = list(values) if isinstance(values, (list, tuple)) else parse_design_values(values)
        selected = []
        for value in requested:
            matches = std_indvarmatch(value, factor_values)
            if matches:
                selected.append(matches[0])
        if not selected and factor_values:
            selected = [1]
        return factor_index, selected, 1
    if numerical:
        return factor_index, list(range(1, len(factor_values) + 1)), 2
    return factor_index, ([1] if factor_values else []), 1


def _encoded_values(values: list[Any]) -> list[str]:
    return [encode_design_value(value, separator=_VALUE_SEPARATOR) for value in values]


def _fill_value_list(widget: Any, values: list[Any], selected_rows: list[int] | None) -> None:
    widget.clear()
    widget.addItems(_encoded_values(values))
    rows = list(range(widget.count())) if selected_rows is None else selected_rows
    for row in rows:
        if 0 <= row < widget.count():
            widget.item(row).setSelected(True)
    if widget.count() and rows:
        widget.setCurrentRow(rows[0])


def _values_are_numeric(values: list[Any]) -> bool:
    return not values or all(_is_number(value) for value in values)


def _is_number(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _unique_values(values: list[Any]) -> list[Any]:
    try:
        return sorted(set(values))
    except TypeError:
        unique: list[Any] = []
        for value in values:
            if value not in unique:
                unique.append(value)
        return unique


__all__ = [
    "apply_addindepvar_combine",
    "apply_addindepvar_selectfact",
    "apply_addindepvar_vartype",
    "encode_design_value",
    "normalize_addindepvar_varlist",
    "pop_addindepvar",
    "pop_addindepvar_dialog_spec",
    "read_addindepvar_selection",
    "selection_from_addindepvar_result",
    "with_none_factor",
]
