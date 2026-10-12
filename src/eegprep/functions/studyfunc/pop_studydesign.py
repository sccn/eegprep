"""Create, select, and edit STUDY designs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.guifunc.inputdlg2 import inputdlg2
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
from eegprep.functions.studyfunc.pop_listfactors import pop_listfactors
from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy
from eegprep.functions.studyfunc.std_addvarlevel import std_addvarlevel
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.studyfunc.std_getindvar import std_getindvar
from eegprep.functions.studyfunc.std_limodesign import std_limodesign
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_maketrialinfo import std_maketrialinfo
from eegprep.functions.studyfunc.std_selectdesign import std_selectdesign


# EEGLAB's std_makedesign command stores variable1 through variable4. The dialog
# itself appends every New variable; OK keeps the first four.
_STORED_DESIGN_VARIABLES = 4
_FIRST_DESIGN_WARNING = "The first STUDY design cannot be removed, only modified"
_UNKNOWN_SUBJECT_WARNING = "Unknown subject(s) - check syntax"


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

    The dialog edits every current design: subjects, design names, independent
    variables (including a continuous factor such as ``rt``), and optional
    resave. Trial info is rebuilt when a chosen factor is not already available.
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
        editor = _editor_from_result(result)
        if editor is not None:
            _apply_pending_subjects(editor, str(result.get("subjects") or ""))
            study, datasets, command = commit_studydesign_editor(
                study,
                datasets,
                editor,
                resave=bool(result.get("resave")),
            )
            return (study, datasets, command) if return_com else (study, datasets)
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
    state = studydesign_editor_state(study, datasets)
    current = int(state["design_index"])
    design_names = [
        str(item.get("name") or f"STUDY.design {index}") for index, item in enumerate(state["designs"], start=1)
    ]
    variable_summary = variable_summary_lines(state["variables"])
    controls = [
        ControlSpec("text", "Include these subjects (default: all)", font_weight="bold"),
        ControlSpec(
            "edit",
            tag="subjects",
            value=state["subject_text"],
            callback=CallbackSpec("studydesign_set_subjects", state),
        ),
        ControlSpec(
            "pushbutton",
            "...",
            tag="select_subjects",
            callback=CallbackSpec("studydesign_select_subjects", state),
        ),
        ControlSpec("text", "Design name", font_weight="bold"),
        ControlSpec("spacer"),
        ControlSpec("pushbutton", "New", tag="new_design", callback=CallbackSpec("studydesign_add_design", state)),
        ControlSpec(
            "pushbutton",
            "Rename",
            tag="rename_design",
            callback=CallbackSpec("studydesign_rename_design", state),
        ),
        ControlSpec(
            "pushbutton",
            "Delete",
            tag="delete_design",
            callback=CallbackSpec("studydesign_delete_design", state),
        ),
        ControlSpec(
            "listbox",
            "|".join(design_names or [f"STUDY.design {current}"]),
            tag="designind",
            value=current,
            callback=CallbackSpec("studydesign_select_design", state),
        ),
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
            callback=CallbackSpec("studydesign_new_variable", state),
        ),
        ControlSpec(
            "pushbutton",
            "Edit",
            tag="edit_variable",
            callback=CallbackSpec("studydesign_edit_variable", state),
        ),
        ControlSpec(
            "pushbutton",
            "Delete",
            tag="delete_variable",
            callback=CallbackSpec("studydesign_delete_variable", state),
        ),
        ControlSpec(
            "pushbutton",
            "List factors",
            tag="list_factors",
            callback=CallbackSpec("studydesign_list_factors", state),
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
            "Web help opens the packaged pop_studydesign help. EEGLAB opens the STUDY design wiki page.",
            "Paired-statistics checkboxes, Plot, data-trial selection, and the design folder field are commented out of the current EEGLAB dialog.",
            "std_makedesign stores variable1 through variable4. Additional dialog variables are dropped on OK, as in EEGLAB.",
            "An empty design records variable1='none' and variable2='none'. EEGLAB omits those labels, and re-running its history restores condition.",
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
        for index, variable in enumerate(variables[:_STORED_DESIGN_VARIABLES], start=1):
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


def studydesign_editor_state(study: dict[str, Any], datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the mutable design-editor state shared by the dialog callbacks."""
    designs = [deepcopy(design) for design in study.get("design") or [] if isinstance(design, dict)]
    if not designs:
        designs = [_blank_design(1, list(study.get("subject") or []))]
    current = int(study.get("currentdesign") or 1)
    current = min(max(current, 1), len(designs))
    _ensure_variable_list(designs[current - 1])
    state = {
        "designs": designs,
        "subjects": [str(item) for item in study.get("subject") or []],
        "datasetinfo": deepcopy(study.get("datasetinfo") or []),
        "varlist": studydesign_varlist(study, datasets),
        "design_index": current,
        "variable_index": 1,
        "subject_text": "",
    }
    _sync_variable_alias(state)
    state["subject_text"] = displayed_subject_text(state) or ""
    return state


def displayed_subject_text(state: dict[str, Any]) -> str | None:
    """Return the subject box text EEGLAB writes when the design uses a subset."""
    design = state["designs"][int(state["design_index"]) - 1]
    values = list((design.get("cases") or {}).get("value") or [])
    if len(values) == len(state["subjects"]):
        return None
    return "".join(f"{item} " for item in values)


def apply_studydesign_set_subjects(state: dict[str, Any], text: str) -> str | None:
    """Apply the subject edit box to every design, matching ``setsubj``."""
    tokens = parse_subject_text(text)
    if any(token not in state["subjects"] for token in tokens):
        for design in state["designs"]:
            cases = design.setdefault("cases", {"label": "subject", "value": []})
            cases["value"] = list(state["subjects"])
        state["subject_text"] = ""
        return _UNKNOWN_SUBJECT_WARNING
    for design in state["designs"]:
        cases = design.setdefault("cases", {"label": "subject", "value": []})
        cases["label"] = cases.get("label") or "subject"
        cases["value"] = tokens
    state["subject_text"] = text
    return None


def apply_studydesign_select_subjects(state: dict[str, Any], labels: list[str]) -> str:
    """Store a subject-picker selection on every design. Return the edit-box text."""
    for design in state["designs"]:
        cases = design.setdefault("cases", {"label": "subject", "value": []})
        cases["label"] = cases.get("label") or "subject"
        cases["value"] = list(labels)
    if len(labels) == len(state["subjects"]):
        state["subject_text"] = ""
    else:
        state["subject_text"] = " ".join(labels)
        if state["subject_text"]:
            state["subject_text"] += " "
    return state["subject_text"]


def apply_studydesign_add(state: dict[str, Any]) -> None:
    """Copy the current design, clear its variables, and select the copy."""
    source = deepcopy(state["designs"][int(state["design_index"]) - 1])
    source["variable"] = []
    source["limo"] = []
    source["name"] = f"Design {len(state['designs']) + 1}"
    state["designs"].append(source)
    state["design_index"] = len(state["designs"])
    state["variable_index"] = 1
    _sync_variable_alias(state)


def apply_studydesign_delete(state: dict[str, Any]) -> str | None:
    """Delete the selected design. The first design cannot be removed."""
    if int(state["design_index"]) == 1:
        return _FIRST_DESIGN_WARNING
    del state["designs"][int(state["design_index"]) - 1]
    state["design_index"] = min(int(state["design_index"]), len(state["designs"]))
    state["variable_index"] = 1
    _sync_variable_alias(state)
    return None


def apply_studydesign_rename(state: dict[str, Any], name: str) -> None:
    """Rename the selected design."""
    state["designs"][int(state["design_index"]) - 1]["name"] = name


def apply_studydesign_select(state: dict[str, Any], index: int) -> None:
    """Select a 1-based design and show its variables."""
    if index < 1 or index > len(state["designs"]):
        return
    state["design_index"] = index
    state["variable_index"] = 1
    _sync_variable_alias(state)


def apply_studydesign_delete_variable(state: dict[str, Any]) -> bool:
    """Delete the selected independent variable."""
    variables = state["variables"]
    index = int(state.get("variable_index") or 0) - 1
    if index < 0 or index >= len(variables):
        return False
    del variables[index]
    state["variable_index"] = min(index + 1, len(variables)) or 1
    _remember_variables(state)
    return True


def run_studydesign_rename(state: dict[str, Any], *, renderer: Any) -> bool:
    """Open the rename prompt. Return whether the name changed."""
    design = state["designs"][int(state["design_index"]) - 1]
    result = inputdlg2(
        ["Study design name:                                                                    "],
        "Rename Study Design",
        1,
        [str(design.get("name") or "")],
        "pop_studydesign",
        renderer=renderer,
    )
    if not result:
        return False
    apply_studydesign_rename(state, result[0])
    return True


def run_studydesign_edit_variable(state: dict[str, Any], *, renderer: Any) -> bool:
    """Replace the selected variable from Add variable. Return whether it changed."""
    variables = state["variables"]
    index = int(state.get("variable_index") or 0) - 1
    if index < 0 or index >= len(variables):
        return False
    variable = variables[index]
    values = [] if str(variable.get("vartype") or "").lower() == "continuous" else variable.get("value")
    result = inputgui(
        pop_addindepvar_dialog_spec(state["varlist"], var=str(variable.get("label") or ""), values=values),
        renderer=renderer,
    )
    if result is None:
        return False
    label, picked, categorical = selection_from_addindepvar_result(state["varlist"], result)
    if not label or label == "None":
        return False
    variable["label"] = label
    variable["value"] = [] if not categorical else list(picked)
    variable["vartype"] = "categorical" if categorical else "continuous"
    _remember_variables(state)
    return True


def run_studydesign_new_variable(
    params: dict[str, Any], *, renderer: Any
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Open Add variable and append the chosen factor to the dialog's design variables."""
    result = inputgui(pop_addindepvar_dialog_spec(params["varlist"]), renderer=renderer)
    if result is None:
        return None, None
    updated, warning = apply_studydesign_new_variable(params["variables"], params["varlist"], result)
    if warning or updated is params["variables"]:
        return warning, None
    params["variables"] = updated
    params["variable_index"] = len(updated)
    _remember_variables(params)
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
    updated = [dict(variable) for variable in variables]
    updated.append(
        {
            "label": label,
            "value": [] if not categorical else list(values),
            "vartype": "categorical" if categorical else "continuous",
        }
    )
    return updated, None


def studydesign_factor_lines(design: dict[str, Any], datasetinfo: list[dict[str, Any]]) -> list[str]:
    """Return the List factors window text for one design."""
    study = std_addvarlevel(
        {
            "datasetinfo": deepcopy(datasetinfo),
            "design": [deepcopy(design)],
            "subject": [],
            "name": "",
        },
        1,
    )
    leveled = study["design"][0]
    factors = pop_listfactors(leveled, constant="off")
    if not isinstance(factors, list):
        factors = []
    one = [factor for factor in factors if str(factor.get("level") or "one") == "one"]
    two = [factor for factor in factors if str(factor.get("level") or "") == "two"]
    _cat1, _cont1, des1 = std_limodesign(one, [], desconly="on")
    _cat2, _cont2, des2 = std_limodesign(two, [], desconly="on")
    lines: list[str] = []
    if _design_has_terms(des1):
        lines.extend(("1st-level variables", "(intra subject)", *_level_one_labels(des1)))
    if _design_has_terms(des2):
        lines.extend(("2nd-level available statistics", "(inter subject)", *_level_two_labels(des1, des2)))
    return lines


def list_factors_dialog_spec(lines: list[str]) -> DialogSpec:
    """Return the read-only List factors dialog."""
    controls = tuple(ControlSpec("text", line or " ") for line in lines) or (ControlSpec("text", "No factors"),)
    return DialogSpec(
        title="List factors",
        controls=controls,
        geometry=tuple((1,) for _control in controls),
        function_name="pop_listfactors",
        eeglab_source="functions/studyfunc/pop_listfactors.m",
        help_text=None,
        show_help_button=False,
        size=(420, min(640, 80 + 22 * len(controls))),
        known_differences=(
            "EEGLAB closes this window with a single OK button. EEGPrep also shows Cancel; neither changes the STUDY.",
        ),
    )


def commit_studydesign_editor(
    study: dict[str, Any],
    datasets: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    resave: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Write the editor state with the same command sequence EEGLAB runs on OK."""
    original = [deepcopy(design) for design in study.get("design") or []]
    designs = state["designs"]
    commands: list[str] = []
    while len(study.get("design") or []) > len(designs):
        removed = len(study["design"])
        del study["design"][-1]
        commands.append(f"del STUDY['design'][{removed - 1}]")
    current_after = int(study.get("currentdesign") or 1)
    for index, design in enumerate(designs, start=1):
        if index <= len(original) and _same_design(original[index - 1], design):
            continue
        options = _options_from_editor_design(design)
        study, prefix = _ensure_trialinfo(study, datasets, options)
        if prefix:
            commands.append(prefix.rstrip("\n"))
        study, command = std_makedesign(study, datasets, index, return_com=True, **options)
        commands.append(command)
        current_after = index
    selected = int(state["design_index"])
    if selected != current_after:
        study = std_selectdesign(study, datasets, selected)
        commands.append(build_python_call(("STUDY",), "std_selectdesign", "STUDY", "ALLEEG", str(selected)))
    if resave and study.get("filename"):
        study, command = pop_savestudy(study, datasets, savemode="resave", return_com=True)
        commands.append(command)
    return study, datasets, "\n".join(command for command in commands if command)


def parse_subject_text(text: str) -> list[str]:
    """Split the subject box the way EEGLAB's ``textscan`` does."""
    return [token.strip("'") for token in str(text).split() if token.strip("'")]


def sync_studydesign_widgets(widgets: dict[str, Any], state: dict[str, Any]) -> None:
    """Refresh the design list, subject box, and variable list from ``state``."""
    design_list = widgets["designind"]
    design_list.blockSignals(True)
    design_list.clear()
    design_list.addItems([str(design.get("name") or "") for design in state["designs"]])
    design_list.setCurrentRow(int(state["design_index"]) - 1)
    design_list.blockSignals(False)
    design_list.setProperty("eegprep_value", state)

    subject_text = displayed_subject_text(state)
    subjects = widgets["subjects"]
    subjects.blockSignals(True)
    if subject_text is not None:
        subjects.setText(subject_text)
        state["subject_text"] = subject_text
    elif state.get("subject_text") == "":
        subjects.setText("")
    subjects.blockSignals(False)

    summary = widgets["variable_summary"]
    rows = variable_summary_lines(state["variables"])
    summary.blockSignals(True)
    summary.clear()
    summary.addItems(rows or ["No independent variables"])
    row = int(state.get("variable_index") or 1) - 1
    if rows and 0 <= row < summary.count():
        summary.setCurrentRow(row)
    summary.blockSignals(False)
    summary.setProperty("eegprep_value", list(state["variables"]))


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


def _editor_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    editor = result.get("designind")
    if isinstance(editor, dict) and isinstance(editor.get("designs"), list):
        return editor
    return None


def _apply_pending_subjects(state: dict[str, Any], text: str) -> None:
    if text == str(state.get("subject_text") or ""):
        return
    apply_studydesign_set_subjects(state, text)


def _ensure_variable_list(design: dict[str, Any]) -> list[dict[str, Any]]:
    variables = design.get("variable")
    if not isinstance(variables, list):
        variables = []
        design["variable"] = variables
    return variables


def _sync_variable_alias(state: dict[str, Any]) -> None:
    design = state["designs"][int(state["design_index"]) - 1]
    state["variables"] = _ensure_variable_list(design)


def _remember_variables(state: dict[str, Any]) -> None:
    design = state["designs"][int(state["design_index"]) - 1]
    design["variable"] = state["variables"]
    preview = deepcopy(state["varlist"].get("preview") or {})
    preview["design"] = deepcopy(state["designs"])
    preview["datasetinfo"] = state["datasetinfo"]
    state["varlist"] = _varlist_from_preview(preview)


def _blank_design(index: int, subjects: list[str]) -> dict[str, Any]:
    return {
        "name": f"STUDY.design {index}",
        "filepath": "",
        "variable": [],
        "cases": {"label": "subject", "value": list(subjects)},
        "include": [],
    }


def _options_from_editor_design(design: dict[str, Any]) -> dict[str, Any]:
    labeled = [
        variable for variable in design.get("variable") or [] if isinstance(variable, dict) and variable.get("label")
    ]
    options: dict[str, Any] = {
        "name": str(design.get("name") or ""),
        "defaultdesign": "off",
    }
    if not labeled:
        options["variable1"] = "none"
        options["variable2"] = "none"
    for index, variable in enumerate(labeled[:_STORED_DESIGN_VARIABLES], start=1):
        vartype = str(variable.get("vartype") or "categorical")
        options[f"variable{index}"] = str(variable["label"])
        options[f"vartype{index}"] = vartype
        options[f"values{index}"] = [] if vartype == "continuous" else list(variable.get("value") or [])
    cases = list((design.get("cases") or {}).get("value") or [])
    if cases:
        options["subjselect"] = cases
    if design.get("filepath"):
        options["filepath"] = str(design["filepath"])
    if design.get("include"):
        options["datselect"] = design["include"]
    return options


def _same_design(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _plain_value(left) == _plain_value(right)


def _plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _design_has_terms(design: dict[str, Any]) -> bool:
    return bool(design.get("categorical") or design.get("continuous"))


def _format_condition(pairs: list[Any]) -> str:
    parts = []
    for pair in pairs:
        label, value = pair
        if value in ("", None):
            parts.append(f"{label} (continuous)")
        elif isinstance(value, bool):
            parts.append(f"{label} = {value}")
        elif isinstance(value, (int, float)):
            number = float(value)
            parts.append(f"{label} = {int(number) if number.is_integer() else number:g}")
        elif isinstance(value, (list, tuple)):
            parts.append(str(label))
        else:
            parts.append(f"{label} = {value}")
    return " & ".join(parts)


def _level_one_labels(design: dict[str, Any]) -> list[str]:
    lines = []
    count = 1
    for group in design.get("categorical") or []:
        for option in group:
            lines.append(f"{count}. {_format_condition(option)}")
            count += 1
    for option in design.get("continuous") or []:
        lines.append(f"{count}. {_format_condition(option)}")
        count += 1
    lines.append(f"{count}. Constant")
    return lines


def _level_two_labels(level_one: dict[str, Any], level_two: dict[str, Any]) -> list[str]:
    lines = ["* One-sample t-test on 1st level var."]
    if level_one.get("categorical"):
        lines.append("* Paire/unpaired t-test on 1st level var.")
    if any(len(group) > 2 for group in level_one.get("categorical") or []):
        lines.append("* ANOVA on 1st level var.")
    if len(level_one.get("categorical") or []) > 1 and len(level_one.get("continuous") or []) > 1:
        lines.append("* ANCOVA on 1st level var.")
    continuous = level_two.get("continuous") or []
    for group in level_two.get("categorical") or []:
        for option in group:
            kind = "t-test/ANOVA/ANCOVA" if continuous else "t-test/ANOVA"
            lines.append(f"* {kind} using {_format_condition(option)}")
    for option in continuous:
        label = option[0][0] if option else ""
        lines.append(
            f"* Regression/ANCOVA using {label}" if continuous else f"* Regression using {_format_condition(option)}"
        )
    lines.append("* Regression/ANOVA/ANCOVA with text file")
    lines.append("  (text file must have 1 row per subject)")
    if len(lines) > 15:
        lines = lines[:15]
        lines.append("...")
    return lines


def _ensure_trialinfo(
    study: dict[str, Any], datasets: list[dict[str, Any]], options: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    labels = [
        str(options[key])
        for key in sorted(options)
        if key.startswith("variable")
        and key.removeprefix("variable").isdigit()
        and str(options[key]).strip().lower() not in {"", "none", "[]"}
    ]
    if not labels or all(label in available_variables(study) for label in labels):
        return study, ""
    study, _rows = std_maketrialinfo(study, datasets)
    return study, "STUDY, alltrialinfo = std_maketrialinfo(STUDY, ALLEEG)\n"


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
    "apply_studydesign_add",
    "apply_studydesign_delete",
    "apply_studydesign_delete_variable",
    "apply_studydesign_rename",
    "apply_studydesign_select_subjects",
    "apply_studydesign_set_subjects",
    "commit_studydesign_editor",
    "list_factors_dialog_spec",
    "pop_studydesign",
    "pop_studydesign_dialog_spec",
    "run_studydesign_new_variable",
    "studydesign_editor_state",
    "studydesign_factor_lines",
    "studydesign_varlist",
    "variable_summary_lines",
]
