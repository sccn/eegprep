"""Create, select, and edit STUDY designs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._study_utils import (
    available_variables,
    build_python_call,
    ensure_study,
    parse_design_values,
)
from eegprep.functions.studyfunc.pop_addindepvar import (
    encode_design_value,
    pop_addindepvar_dialog_spec,
    selection_from_addindepvar_result,
    with_none_factor,
)
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.studyfunc.std_getindvar import std_getindvar
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_maketrialinfo import std_maketrialinfo
from eegprep.functions.studyfunc.std_selectdesign import std_selectdesign


_MAX_DESIGN_VARIABLES = 4


def pop_studydesign(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    designind: int | None = None,
    *args: Any,
    gui: bool | str | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Edit STUDY designs and select the current design.

    The GUI New button under independent variables opens Add variable. A
    continuous factor such as ``rt`` is stored on ``STUDY.design``. Trial info
    is rebuilt when that factor is not already available.
    """
    study, datasets = std_checkset(ensure_study(STUDY), ALLEEG)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    gui = options.pop("gui", gui)
    if designind is None:
        designind = int(options.pop("designind", study.get("currentdesign") or 1))
    use_gui = is_on(gui) if gui is not None else False
    if use_gui:
        result = inputgui(pop_studydesign_dialog_spec(study, datasets), renderer=renderer)
        if result is None:
            return (study, datasets, "") if return_com else (study, datasets)
        options.update(_options_from_gui(result, study, int(designind)))
        designind = int(result.get("designind") or designind or 1)

    if options:
        study, prefix = _ensure_trialinfo(study, datasets, options)
        study, command = std_makedesign(study, datasets, int(designind), return_com=True, **options)
        if prefix:
            command = f"{prefix}{command}"
    else:
        study = std_selectdesign(study, datasets, int(designind))
        command = build_python_call(
            ("STUDY",),
            "std_selectdesign",
            "STUDY",
            "ALLEEG",
            str(int(designind)),
        )
    return (study, datasets, command) if return_com else (study, datasets)


def pop_studydesign_dialog_spec(STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None) -> DialogSpec:
    """Return the EEGLAB-like STUDY design dialog spec."""
    study, datasets = std_checkset(ensure_study(STUDY), ALLEEG)
    designs = study.get("design") or []
    current = int(study.get("currentdesign") or 1)
    design = designs[current - 1] if 0 < current <= len(designs) else {}
    variables = _editable_variables(design.get("variable") or [])
    design_names = [str(item.get("name") or f"STUDY.design {index}") for index, item in enumerate(designs, start=1)]
    variable_summary = variable_summary_lines(variables)
    varlist = studydesign_varlist(study, datasets)
    coming_soon = CallbackSpec(
        "show_message",
        {
            "title": "STUDY design editor",
            "message": (
                "This EEGLAB subdialog is not available yet in EEGPrep. "
                "Use pop_studydesign(..., variable1=..., values1=...) from the console for now."
            ),
        },
    )
    controls = [
        ControlSpec("text", "Include these subjects (default: all)", font_weight="bold"),
        ControlSpec("edit", tag="subjects", value=""),
        ControlSpec(
            "pushbutton",
            "...",
            tag="select_subjects",
            callback=_button_callback(coming_soon, "select_subjects"),
        ),
        ControlSpec("text", "Design name", font_weight="bold"),
        ControlSpec("spacer"),
        ControlSpec("pushbutton", "New", tag="new_design", callback=_button_callback(coming_soon, "new_design")),
        ControlSpec(
            "pushbutton", "Rename", tag="rename_design", callback=_button_callback(coming_soon, "rename_design")
        ),
        ControlSpec(
            "pushbutton", "Delete", tag="delete_design", callback=_button_callback(coming_soon, "delete_design")
        ),
        ControlSpec("listbox", "|".join(design_names or [f"STUDY.design {current}"]), tag="designind", value=current),
        ControlSpec(
            "text",
            "Edit the independent variables for this design",
            tag="independent_header",
            font_weight="bold",
        ),
        ControlSpec("spacer"),
        ControlSpec(
            "pushbutton",
            "New",
            tag="new_variable",
            callback=CallbackSpec(
                "studydesign_new_variable",
                {
                    "button": "new_variable",
                    "varlist": varlist,
                    "variables": variables,
                    "design_index": current,
                },
            ),
        ),
        ControlSpec("pushbutton", "Edit", tag="edit_variable", callback=_button_callback(coming_soon, "edit_variable")),
        ControlSpec(
            "pushbutton", "Delete", tag="delete_variable", callback=_button_callback(coming_soon, "delete_variable")
        ),
        ControlSpec(
            "pushbutton", "List factors", tag="list_factors", callback=_button_callback(coming_soon, "list_factors")
        ),
        ControlSpec(
            "listbox", "|".join(variable_summary or ["No independent variables"]), tag="variable_summary", value=1
        ),
        ControlSpec("checkbox", " Re-save STUDY file (if saved previously)", tag="resave", value=True),
    ]
    return DialogSpec(
        title="Edit STUDY design -- pop_studydesign()",
        controls=tuple(controls),
        geometry=(
            (2.5, 1.5, 0.45),
            (1,),
            (2.8, 0.45, 0.6, 0.6),
            (1,),
            (1,),
            (1.6, 0.45, 0.45, 0.55, 0.8),
            (1,),
            (1,),
        ),
        function_name="pop_studydesign",
        eeglab_source="functions/studyfunc/pop_studydesign.m",
        help_text="pop_studydesign",
        help_label="Web help",
        size=(484, 568),
        content_margins=(24, 22, 24, 14),
        row_spacing=5,
        geomvert=(1, 0.65, 0.01, 3.1, 0.65, 0.01, 3.4, 1),
        button_size=(86, 22),
        extra_stylesheet="""
            QDialog#pop_studydesign QLabel,
            QDialog#pop_studydesign QCheckBox,
            QDialog#pop_studydesign QPushButton,
            QDialog#pop_studydesign QLineEdit,
            QDialog#pop_studydesign QListWidget {
                font-size: 12px;
            }
            QDialog#pop_studydesign QLineEdit,
            QDialog#pop_studydesign QPushButton {
                min-height: 20px;
                max-height: 20px;
            }
            QDialog#pop_studydesign QListWidget#designind {
                min-height: 90px;
                max-height: 90px;
            }
            QDialog#pop_studydesign QLabel#independent_header {
                border: 1px solid #7f7f7f;
                border-bottom: 0;
                padding-left: 4px;
            }
            QDialog#pop_studydesign QListWidget#variable_summary {
                min-height: 105px;
                max-height: 105px;
            }
            QDialog#pop_studydesign QPushButton#select_subjects,
            QDialog#pop_studydesign QPushButton#new_design,
            QDialog#pop_studydesign QPushButton#rename_design,
            QDialog#pop_studydesign QPushButton#delete_design,
            QDialog#pop_studydesign QPushButton#new_variable,
            QDialog#pop_studydesign QPushButton#edit_variable,
            QDialog#pop_studydesign QPushButton#delete_variable {
                min-width: 54px;
                max-width: 54px;
                padding: 0 3px;
            }
            QDialog#pop_studydesign QPushButton#list_factors {
                min-width: 82px;
                max-width: 82px;
                padding: 0 3px;
            }
        """,
        known_differences=(
            "The New button under independent variables opens Add variable, including continuous trial factors such as rt.",
            "Subject selection, design New/Rename/Delete, Edit, Delete variable, and List factors still show an unavailable message.",
            "A design stores at most four independent variables, matching std_makedesign.",
        ),
    )


def _options_from_gui(
    result: dict[str, Any], study: dict[str, Any] | None = None, designind: int = 1
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if "name" in result:
        options["name"] = str(result.get("name") or "")
    for index in range(1, 5):
        variable_key = f"variable{index}"
        values_key = f"values{index}"
        if variable_key in result and not isinstance(result.get(variable_key), dict):
            options[variable_key] = str(result.get(variable_key) or "")
        if values_key in result and not isinstance(result.get(values_key), dict):
            options[values_key] = parse_design_values(result.get(values_key))
    variables = result.get("variable_summary")
    if _is_variable_list(variables):
        design = _selected_design(study or {}, int(result.get("designind") or designind or 1))
        if design.get("name"):
            options["name"] = str(design["name"])
        for index, variable in enumerate(variables[:_MAX_DESIGN_VARIABLES], start=1):
            options[f"variable{index}"] = str(variable["label"])
            if str(variable.get("vartype") or "categorical") == "continuous":
                options[f"values{index}"] = []
            else:
                options[f"values{index}"] = list(variable.get("value") or [])
            options[f"vartype{index}"] = str(variable.get("vartype") or "categorical")
    subjects = parse_design_values(result.get("subjects"))
    if subjects:
        options["subjselect"] = subjects
    return options


def variable_summary_lines(variables: list[dict[str, Any]]) -> list[str]:
    """Return the independent-variable list text for the design dialog."""
    rows: list[str] = []
    for variable in variables:
        if not isinstance(variable, dict):
            continue
        label = str(variable.get("label") or "")
        if not label:
            continue
        values = variable.get("value") or []
        vartype = str(variable.get("vartype") or "categorical").lower()
        if values and vartype == "categorical":
            encoded = [encode_design_value(value) for value in values]
            rows.append(f"Categorical variable: {label} - Values ({' - '.join(encoded)})")
        else:
            rows.append(f"Continuous variable: {label}")
    return rows


def studydesign_varlist(study: dict[str, Any], datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Return Add variable factors, including trial fields such as ``rt``."""
    preview = {
        "datasetinfo": deepcopy(study.get("datasetinfo") or []),
        "design": deepcopy(study.get("design") or []),
        "subject": list(study.get("subject") or []),
        "name": study.get("name", ""),
    }
    if _trialinfo_absent(preview) and any(int(eeg.get("trials") or 1) > 1 for eeg in datasets):
        sources = [_trialinfo_source(eeg) for eeg in datasets]
        preview, _rows = std_maketrialinfo(preview, sources)
    return _varlist_from_preview(preview)


def run_studydesign_new_variable(
    params: dict[str, Any], *, renderer: Any
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Open Add variable and append the chosen factor to the dialog's design variables."""
    result = inputgui(pop_addindepvar_dialog_spec(params["varlist"]), renderer=renderer)
    if result is None:
        return None, None
    updated, warning = apply_studydesign_new_variable(params["variables"], params["varlist"], result)
    if warning:
        return warning, None
    if updated is params["variables"]:
        return None, None
    params["variables"] = updated
    params["varlist"] = refresh_studydesign_varlist(params["varlist"], updated, int(params["design_index"]))
    return None, updated


def apply_studydesign_new_variable(
    variables: list[dict[str, Any]],
    varlist: dict[str, Any],
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    """Append one Add variable selection, matching ``pop_studydesign('newvar')``."""
    label, values, categorical = selection_from_addindepvar_result(varlist, result)
    if not label or label == "None":
        return variables, None
    if len(variables) >= _MAX_DESIGN_VARIABLES:
        return variables, "A STUDY design supports at most 4 independent variables."
    updated = [dict(variable) for variable in variables]
    updated.append(
        {
            "label": label,
            "value": [] if not categorical else list(values),
            "vartype": "categorical" if categorical else "continuous",
        }
    )
    return updated, None


def refresh_studydesign_varlist(
    varlist: dict[str, Any], variables: list[dict[str, Any]], design_index: int
) -> dict[str, Any]:
    """Rebuild factor choices after a design variable is added."""
    preview = deepcopy(varlist.get("preview") or {})
    designs = list(preview.get("design") or [])
    while len(designs) < design_index:
        designs.append({"name": f"STUDY.design {len(designs) + 1}", "variable": []})
    if designs:
        current = dict(designs[design_index - 1])
        current["variable"] = [dict(variable) for variable in variables]
        designs[design_index - 1] = current
    preview["design"] = designs
    return _varlist_from_preview(preview)


def _button_callback(template: CallbackSpec, button: str) -> CallbackSpec:
    params = dict(template.params)
    params["button"] = button
    return CallbackSpec(template.name, params, template.matlab_callback)


def _ensure_trialinfo(
    study: dict[str, Any], datasets: list[dict[str, Any]], options: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    labels = [
        str(options[key])
        for key in sorted(options)
        if key.startswith("variable") and key.removeprefix("variable").isdigit() and options[key]
    ]
    if not labels or all(label in available_variables(study) for label in labels):
        return study, ""
    study, _rows = std_maketrialinfo(study, datasets)
    return study, "STUDY, alltrialinfo = std_maketrialinfo(STUDY, ALLEEG)\n"


def _editable_variables(variables: list[Any]) -> list[dict[str, Any]]:
    editable = []
    for variable in variables:
        if not isinstance(variable, dict) or not variable.get("label"):
            continue
        editable.append(
            {
                "label": str(variable["label"]),
                "value": list(variable.get("value") or []),
                "vartype": str(variable.get("vartype") or "categorical"),
            }
        )
    return editable


def _is_variable_list(value: Any) -> bool:
    return (
        isinstance(value, list) and bool(value) and all(isinstance(item, dict) and item.get("label") for item in value)
    )


def _selected_design(study: dict[str, Any], designind: int) -> dict[str, Any]:
    designs = study.get("design") or []
    if 0 < designind <= len(designs) and isinstance(designs[designind - 1], dict):
        return designs[designind - 1]
    return {}


def _varlist_from_preview(preview: dict[str, Any]) -> dict[str, Any]:
    factors, factorvals, _subjects, _paired = std_getindvar(preview, "both", True)
    return with_none_factor({"factors": factors, "factorvals": factorvals, "preview": preview})


def _trialinfo_absent(study: dict[str, Any]) -> bool:
    infos = [info for info in study.get("datasetinfo") or [] if isinstance(info, dict)]
    if not infos:
        return False
    return not any(info.get("trialinfo") for info in infos)


def _trialinfo_source(eeg: dict[str, Any]) -> dict[str, Any]:
    return {
        "trials": eeg.get("trials", 1),
        "pnts": eeg.get("pnts", 0),
        "srate": eeg.get("srate", 1.0),
        "xmin": eeg.get("xmin", 0.0),
        "xmax": eeg.get("xmax", 0.0),
        "event": eeg.get("event"),
        "epoch": eeg.get("epoch"),
        "trialinfo": eeg.get("trialinfo"),
        "subject": eeg.get("subject", ""),
        "setname": eeg.get("setname", ""),
        "filename": eeg.get("filename", ""),
        "filepath": eeg.get("filepath", ""),
        "condition": eeg.get("condition", ""),
        "group": eeg.get("group", ""),
        "session": eeg.get("session", ""),
        "run": eeg.get("run", ""),
    }


__all__ = [
    "pop_studydesign",
    "pop_studydesign_dialog_spec",
    "run_studydesign_new_variable",
    "studydesign_varlist",
    "variable_summary_lines",
]
