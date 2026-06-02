"""Create, select, and edit STUDY designs."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._study_utils import (
    as_alleeg_list,
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
    design_names = [str(item.get("name") or f"STUDY.design {index}") for index, item in enumerate(designs, start=1)]
    variable_summary = _variable_summary(variables)
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
        ControlSpec("pushbutton", "New", tag="new_variable", callback=_button_callback(coming_soon, "new_variable")),
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
            "EEGPrep Phase 5a edits and selects design metadata; LIMO, precompute, and plotting hooks are later phases.",
        ),
    )


def _options_from_gui(result: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if "name" in result:
        options["name"] = str(result.get("name") or "")
    for index in range(1, 5):
        variable_key = f"variable{index}"
        values_key = f"values{index}"
        if variable_key in result:
            options[variable_key] = str(result.get(variable_key) or "")
        if values_key in result:
            options[values_key] = parse_design_values(result.get(values_key))
    subjects = parse_design_values(result.get("subjects"))
    if subjects:
        options["subjselect"] = subjects
    return options


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


def _button_callback(template: CallbackSpec, button: str) -> CallbackSpec:
    params = dict(template.params)
    params["button"] = button
    return CallbackSpec(template.name, params, template.matlab_callback)


__all__ = ["pop_studydesign", "pop_studydesign_dialog_spec"]
