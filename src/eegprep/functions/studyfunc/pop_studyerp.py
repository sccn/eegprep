"""Create a simple ERP STUDY."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.studyfunc._study_utils import build_matlab_call, matlab_command_cell
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.std_editset import std_editset


def pop_studyerp(
    ALLEEG: list[dict[str, Any]] | None = None,
    *,
    gui: bool = False,
    renderer: Any = None,
    return_com: bool = False,
) -> Any:
    """Create a STUDY marked as a simple ERP design.

    ``gui=True`` asks for the condition and subject counts, then a dataset
    grid, and builds the STUDY only after Ok. Cancel leaves the workspace
    unchanged.
    """
    if gui or renderer is not None:
        return _studyerp_from_dialogs(renderer, return_com=return_com)
    study, datasets, _command = pop_study(None, ALLEEG or [], name="Simple ERP STUDY", design="ERP", return_com=True)
    command = build_matlab_call(("STUDY", "ALLEEG"), "pop_studyerp", "ALLEEG")
    return (study, datasets, command) if return_com else (study, datasets)


def pop_studyerp_counts_dialog_spec() -> DialogSpec:
    """Return the condition and subject count dialog from ``pop_studyerp.m``."""
    return DialogSpec(
        title="Create a new STUDY set -- pop_studyerp()",
        function_name="pop_studyerp",
        help_text="pop_studyerp",
        eeglab_source="functions/studyfunc/pop_studyerp.m",
        geometry=((1,), (1,), (1, 0.3, 0.4), (1, 0.3, 0.4)),
        controls=(
            ControlSpec("text", "Create simple ERP STUDY", font_weight="bold"),
            ControlSpec(
                "text",
                "This interface creates a simple STUDY. For each subject, trials for each "
                "condition must first be stored in a separate dataset.",
            ),
            ControlSpec("text", "Number of conditions:"),
            ControlSpec("edit", tag="cond", value="1"),
            ControlSpec("spacer"),
            ControlSpec("text", "Number of subjects:"),
            ControlSpec("edit", tag="subjects", value="15"),
            ControlSpec("spacer"),
        ),
    )


def pop_studyerp_grid_dialog_spec(n_conditions: int, n_subjects: int) -> DialogSpec:
    """Return the subjects-by-conditions dataset grid from ``pop_studyerp.m``."""
    controls: list[ControlSpec] = [
        ControlSpec("text", "Create simple ERP STUDY", font_weight="bold"),
        ControlSpec("text", "STUDY set name:"),
        ControlSpec("edit", tag="study_name"),
    ]
    geometry: list[tuple[float, ...] | tuple[int, ...]] = [(1,), (1, 3.5)]
    for index in range(1, n_conditions + 1):
        controls.append(ControlSpec("text", f"Condition {index} name"))
        controls.append(ControlSpec("edit", tag=f"cond{index}"))
        geometry.append((1, 3.5))
    for subject in range(1, n_subjects + 1):
        row: list[float] = []
        for condition in range(1, n_conditions + 1):
            if condition > 1:
                controls.append(ControlSpec("spacer"))
                row.append(0.2)
            controls.append(ControlSpec("edit", tag=f"set{condition}_{subject}"))
            controls.append(
                ControlSpec(
                    "pushbutton",
                    "...",
                    tag=f"browse{condition}_{subject}",
                    callback=CallbackSpec(
                        "select_file",
                        {
                            "button": f"browse{condition}_{subject}",
                            "target": f"set{condition}_{subject}",
                            "caption": "Choose dataset to add to STUDY -- pop_study()",
                            "filter": "EEGLAB dataset (*.set)",
                        },
                    ),
                )
            )
            row.extend((3.5, 0.5))
        geometry.append(tuple(row))
    controls.append(
        ControlSpec(
            "text",
            "When using more than 1 condition, datasets on each line must correspond to the same subject.",
        )
    )
    geometry.append((1,))
    return DialogSpec(
        title="Create a new STUDY set -- pop_studyerp()",
        function_name="pop_studyerp",
        help_text="pop_studyerp",
        eeglab_source="functions/studyfunc/pop_studyerp.m",
        geometry=tuple(geometry),
        controls=tuple(controls),
        scrollable=False,
    )


def _studyerp_from_dialogs(renderer: Any, *, return_com: bool) -> Any:
    counts = inputgui(pop_studyerp_counts_dialog_spec(), renderer=renderer)
    if not counts:
        return (None, [], "") if return_com else (None, [])
    n_conditions = _positive_count(counts.get("cond"), "Number of conditions")
    n_subjects = _positive_count(counts.get("subjects"), "Number of subjects")
    grid = inputgui(pop_studyerp_grid_dialog_spec(n_conditions, n_subjects), renderer=renderer)
    if not grid:
        return (None, [], "") if return_com else (None, [])
    commands = []
    for condition in range(1, n_conditions + 1):
        condition_name = str(grid.get(f"cond{condition}") or "").strip() or f"condition {condition}"
        for subject in range(1, n_subjects + 1):
            filename = str(grid.get(f"set{condition}_{subject}") or "").strip()
            if not filename:
                continue
            commands.append(
                {
                    "index": n_conditions * subject + condition - 1,
                    "load": filename,
                    "subject": f"S{subject:02d}",
                    "condition": condition_name,
                }
            )
    if not commands:
        raise ValueError("pop_studyerp: no dataset files were entered. Fill at least one dataset box, then press Ok.")
    name = str(grid.get("study_name") or "").strip() or "Simple ERP STUDY"
    study, datasets, command = std_editset(
        None,
        [],
        name=name,
        commands=commands,
        updatedat="off",
        return_com=True,
    )
    study["design"] = study.get("design") or [{"name": "ERP"}]
    command = build_matlab_call(
        ("STUDY", "ALLEEG"),
        "std_editset",
        "STUDY",
        "ALLEEG",
        name=name,
        commands=commands,
        updatedat="off",
        dict_formatter=matlab_command_cell,
    )
    return (study, datasets, command) if return_com else (study, datasets)


def _positive_count(value: Any, label: str) -> int:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pop_studyerp: {label} {value!r} is not a whole number. Enter a value such as 1.") from exc
    if number < 1:
        raise ValueError(f"pop_studyerp: {label} must be at least 1. Enter a value such as 1.")
    return number
