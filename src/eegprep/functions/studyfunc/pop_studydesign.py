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
    controls = [
        ControlSpec("text", "Include these subjects (default: all)", font_weight="bold"),
        ControlSpec(
            "edit", tag="subjects", value=" ".join(str(item) for item in design.get("cases", {}).get("value", []))
        ),
        ControlSpec("text", "Design name", font_weight="bold"),
        ControlSpec("edit", tag="name", value=str(design.get("name") or f"STUDY.design {current}")),
        ControlSpec("text", "Current design index"),
        ControlSpec("edit", tag="designind", value=str(current)),
        ControlSpec("text", "Available factors"),
        ControlSpec("edit", tag="available_factors", value=factor_text, enabled=False),
        ControlSpec("text", "Edit the independent variables for this design", font_weight="bold"),
        ControlSpec("text", "Variable 1"),
        ControlSpec("edit", tag="variable1", value=_variable_label(variables, 0)),
        ControlSpec("text", "Values 1 ([] = all)"),
        ControlSpec("edit", tag="values1", value=_variable_values(variables, 0)),
        ControlSpec("text", "Variable 2"),
        ControlSpec("edit", tag="variable2", value=_variable_label(variables, 1)),
        ControlSpec("text", "Values 2 ([] = all)"),
        ControlSpec("edit", tag="values2", value=_variable_values(variables, 1)),
        ControlSpec("text", "Study design factors are taken from STUDY.datasetinfo and trialinfo fields."),
    ]
    return DialogSpec(
        title="Edit STUDY design -- pop_studydesign()",
        controls=tuple(controls),
        geometry=tuple(1 for _control in controls),
        function_name="pop_studydesign",
        eeglab_source="functions/studyfunc/pop_studydesign.m",
        help_text="pop_studydesign",
        size=(620, 430),
        content_margins=(34, 18, 34, 14),
        row_spacing=6,
        known_differences=(
            "EEGPrep Phase 5a edits and selects design metadata; LIMO, precompute, and plotting hooks are later phases.",
        ),
    )


def _options_from_gui(result: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "name": str(result.get("name") or ""),
        "variable1": str(result.get("variable1") or ""),
        "values1": parse_design_values(result.get("values1")),
        "variable2": str(result.get("variable2") or ""),
        "values2": parse_design_values(result.get("values2")),
    }
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


__all__ = ["pop_studydesign", "pop_studydesign_dialog_spec"]
