"""Create, select, and edit STUDY designs."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._study_utils import (
    as_alleeg_list,
    available_variables,
    build_python_call,
    ensure_study,
    parse_design_values,
)
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_selectdesign import std_selectdesign


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
    """Edit STUDY designs and select the current design."""
    datasets = as_alleeg_list(ALLEEG)
    study, datasets = std_checkset(ensure_study(STUDY), datasets)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    gui = options.pop("gui", gui)
    if designind is None:
        designind = int(options.pop("designind", study.get("currentdesign") or 1))
    use_gui = is_on(gui) if gui is not None else False
    if use_gui:
        result = inputgui(pop_studydesign_dialog_spec(study, datasets), renderer=renderer)
        if result is None:
            return (study, datasets, "") if return_com else (study, datasets)
        options.update(_options_from_gui(result))
        designind = int(result.get("designind") or designind or 1)

    if options:
        study, command = std_makedesign(study, datasets, int(designind), return_com=True, **options)
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
    datasets = as_alleeg_list(ALLEEG)
    study, _datasets = std_checkset(ensure_study(STUDY), datasets)
    designs = study.get("design") or []
    current = int(study.get("currentdesign") or 1)
    design = designs[current - 1] if 0 < current <= len(designs) else {}
    variables = design.get("variable") or []
    factors = available_variables(study)
    factor_text = " ".join(factors)
    design_names = [str(item.get("name") or f"STUDY.design {index}") for index, item in enumerate(designs, start=1)]
    variable_summary = _variable_summary(variables)
    controls = [
        ControlSpec("text", "Include these subjects (default: all)", font_weight="bold"),
        ControlSpec(
            "edit", tag="subjects", value=" ".join(str(item) for item in design.get("cases", {}).get("value", []))
        ),
        ControlSpec("pushbutton", "...", tag="select_subjects", enabled=False),
        ControlSpec("text", "Design name", font_weight="bold"),
        ControlSpec("pushbutton", "New", tag="new_design", enabled=False),
        ControlSpec("pushbutton", "Rename", tag="rename_design", enabled=False),
        ControlSpec("pushbutton", "Delete", tag="delete_design", enabled=False),
        ControlSpec("listbox", "|".join(design_names or [f"STUDY.design {current}"]), tag="designind", value=current),
        ControlSpec("text", "Edit the independent variables for this design", font_weight="bold"),
        ControlSpec("pushbutton", "New", tag="new_variable", enabled=False),
        ControlSpec("pushbutton", "Edit", tag="edit_variable", enabled=False),
        ControlSpec("pushbutton", "Delete", tag="delete_variable", enabled=False),
        ControlSpec("pushbutton", "List factors", tag="list_factors", enabled=False),
        ControlSpec(
            "listbox", "|".join(variable_summary or ["No independent variables"]), tag="variable_summary", value=1
        ),
        ControlSpec("text", "Available factors"),
        ControlSpec("edit", tag="available_factors", value=factor_text, enabled=False),
        ControlSpec("text", "Variable 1"),
        ControlSpec("edit", tag="variable1", value=_variable_label(variables, 0)),
        ControlSpec("text", "Values 1 ([] = all)"),
        ControlSpec("edit", tag="values1", value=_variable_values(variables, 0)),
        ControlSpec("text", "Variable 2"),
        ControlSpec("edit", tag="variable2", value=_variable_label(variables, 1)),
        ControlSpec("text", "Values 2 ([] = all)"),
        ControlSpec("edit", tag="values2", value=_variable_values(variables, 1)),
        ControlSpec("checkbox", "Re-save STUDY file (if saved previously)", tag="resave", value=True, enabled=False),
    ]
    return DialogSpec(
        title="Edit STUDY design -- pop_studydesign()",
        controls=tuple(controls),
        geometry=(
            (2.5, 1.5, 0.45),
            (2.9, 0.45, 0.55, 0.55),
            (1,),
            (1,),
            (0.45, 0.45, 0.55, 0.8),
            (1,),
            (1.2, 2.2),
            (0.8, 1.05, 0.9, 1.15),
            (0.8, 1.05, 0.9, 1.15),
            (1,),
        ),
        function_name="pop_studydesign",
        eeglab_source="functions/studyfunc/pop_studydesign.m",
        help_text="pop_studydesign",
        size=(500, 590),
        content_margins=(24, 22, 24, 14),
        row_spacing=5,
        geomvert=(1, 1, 3.1, 0.7, 1, 3.4, 1, 1, 1, 1),
        button_size=(58, 20),
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
                min-height: 86px;
                max-height: 118px;
            }
            QDialog#pop_studydesign QListWidget#variable_summary {
                min-height: 96px;
                max-height: 132px;
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
            "EEGPrep Phase 5a edits and selects design metadata; LIMO, precompute, and plotting hooks are later phases.",
        ),
    )


def _options_from_gui(result: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "variable1": str(result.get("variable1") or ""),
        "values1": parse_design_values(result.get("values1")),
        "variable2": str(result.get("variable2") or ""),
        "values2": parse_design_values(result.get("values2")),
    }
    if "name" in result:
        options["name"] = str(result.get("name") or "")
    subjects = parse_design_values(result.get("subjects"))
    if subjects:
        options["subjselect"] = subjects
    return options


def _variable_label(variables: list[dict[str, Any]], index: int) -> str:
    return str(variables[index].get("label") or "") if index < len(variables) else ""


def _variable_values(variables: list[dict[str, Any]], index: int) -> str:
    if index >= len(variables):
        return ""
    values = variables[index].get("value") or []
    return " ".join(str(value) for value in values)


def _variable_summary(variables: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for variable in variables:
        label = str(variable.get("label") or "")
        if not label:
            continue
        values = variable.get("value") or []
        value_text = " - ".join(str(value) for value in values) if values else "all"
        rows.append(f"Categorical variable: {label} - Values ({value_text})")
    return rows


__all__ = ["pop_studydesign", "pop_studydesign_dialog_spec"]
