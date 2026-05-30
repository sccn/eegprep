"""Create or edit an EEGPrep STUDY from loaded datasets."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._study_utils import (
    as_alleeg_list,
    build_python_call,
    ensure_study,
    parse_optional_int_text,
)
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.studyfunc.std_editset import std_editset
from eegprep.functions.studyfunc.std_makedesign import std_makedesign


def pop_study(
    STUDY: dict[str, Any] | None = None,
    ALLEEG: list[dict[str, Any]] | None = None,
    *args: Any,
    name: str | None = None,
    task: str | None = None,
    notes: str | None = None,
    design: str | None = None,
    gui: bool | str | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Create or edit a STUDY structure from loaded EEG datasets."""
    datasets = as_alleeg_list(ALLEEG)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    gui = options.pop("gui", gui)
    name = options.pop("name", name)
    task = options.pop("task", task)
    notes = options.pop("notes", notes)
    design = options.pop("design", design)
    if options:
        raise ValueError(f"Unknown pop_study option(s): {', '.join(sorted(options))}")
    use_gui = is_on(gui) if gui is not None else False
    study = ensure_study(STUDY)

    if use_gui:
        result = inputgui(pop_study_dialog_spec(study, datasets), renderer=renderer)
        if result is None:
            return (study, datasets, "") if return_com else (study, datasets)
        name = result.get("name", study.get("name", ""))
        task = result.get("task", study.get("task", ""))
        notes = result.get("notes", study.get("notes", ""))
        commands = _dataset_row_commands(datasets, result)
    else:
        commands = None

    if not datasets:
        raise ValueError("pop_study requires at least one loaded dataset")
    study, datasets, _edit_command = std_editset(
        study,
        datasets,
        name=name if name is not None else study.get("name", "EEGPrep study"),
        task=task if task is not None else study.get("task", ""),
        notes=notes if notes is not None else study.get("notes", ""),
        commands=commands,
        return_com=True,
    )
    if not study.get("name"):
        study["name"] = "EEGPrep study"
    if design:
        study, _design_command = std_makedesign(study, datasets, 1, name=design, return_com=True)
    else:
        study, datasets = std_checkset(study, datasets)
    command = build_python_call(
        ("STUDY", "ALLEEG"),
        "pop_study",
        "STUDY",
        "ALLEEG",
        name=study.get("name", ""),
        task=study.get("task", ""),
        notes=study.get("notes", ""),
    )
    return (study, datasets, command) if return_com else (study, datasets)


def pop_study_dialog_spec(STUDY: dict[str, Any] | None, ALLEEG: list[dict[str, Any]] | None) -> DialogSpec:
    """Return the EEGLAB-like STUDY metadata dialog spec."""
    study = ensure_study(STUDY)
    datasets = as_alleeg_list(ALLEEG)
    checked, _datasets = std_checkset(study, datasets)
    title = (
        "Create a new STUDY set -- pop_study()"
        if not STUDY or not STUDY.get("datasetinfo")
        else "Edit STUDY set information - pop_study()"
    )
    controls: list[ControlSpec] = [
        ControlSpec("text", "Create a new STUDY set", font_weight="bold"),
        ControlSpec("text", "STUDY set name:"),
        ControlSpec("edit", tag="name", value=str(checked.get("name") or "")),
        ControlSpec("text", "STUDY set task name:"),
        ControlSpec("edit", tag="task", value=str(checked.get("task") or "")),
        ControlSpec("text", "STUDY set notes:"),
        ControlSpec("edit", tag="notes", value=str(checked.get("notes") or "")),
        ControlSpec("spacer"),
        ControlSpec("text", "dataset filename", font_weight="bold"),
        ControlSpec("text", "subject", font_weight="bold"),
        ControlSpec("text", "session", font_weight="bold"),
        ControlSpec("text", "run", font_weight="bold"),
        ControlSpec("text", "condition", font_weight="bold"),
        ControlSpec("text", "group", font_weight="bold"),
    ]
    datasetinfo = checked.get("datasetinfo") or []
    for index, info in enumerate(datasetinfo, start=1):
        controls.extend(
            (
                ControlSpec("text", str(index)),
                ControlSpec("edit", tag=f"dataset_{index}_filename", value=_display_path(info), enabled=False),
                ControlSpec("edit", tag=f"dataset_{index}_subject", value=str(info.get("subject") or "")),
                ControlSpec(
                    "edit", tag=f"dataset_{index}_session", value=_display_optional_number(info.get("session"))
                ),
                ControlSpec("edit", tag=f"dataset_{index}_run", value=_display_optional_number(info.get("run"))),
                ControlSpec("edit", tag=f"dataset_{index}_condition", value=str(info.get("condition") or "")),
                ControlSpec("edit", tag=f"dataset_{index}_group", value=str(info.get("group") or "")),
            )
        )
    controls.append(
        ControlSpec(
            "text",
            "Important note: Removed datasets will not be saved before being deleted from EEGPrep memory",
        )
    )
    dataset_row_geometry = (0.22, 2.6, 0.9, 0.7, 0.55, 1.3, 0.9)
    geometry = [(1,), (1.2, 5), (1.2, 5), (1.2, 5), dataset_row_geometry]
    geometry.extend(dataset_row_geometry for _info in datasetinfo)
    geometry.append((1,))
    return DialogSpec(
        title=title,
        controls=tuple(controls),
        geometry=tuple(geometry),
        function_name="pop_study",
        eeglab_source="functions/studyfunc/pop_study.m",
        help_text="pop_study",
        size=(850, 560),
        scrollable=len(datasetinfo) > 10,
        content_margins=(34, 18, 34, 14),
        row_spacing=4,
        known_differences=(
            "EEGPrep Phase 5a edits loaded dataset metadata; dataset browsing is provided by pop_studywizard.",
        ),
    )


def _dataset_row_commands(datasets: list[dict[str, Any]], result: dict[str, Any]) -> list[Any]:
    commands: list[Any] = []
    for index, _eeg in enumerate(datasets, start=1):
        prefix = f"dataset_{index}_"
        commands.extend(["index", index])
        if f"{prefix}subject" in result:
            commands.extend(["subject", str(result.get(f"{prefix}subject") or "")])
        if f"{prefix}condition" in result:
            commands.extend(["condition", str(result.get(f"{prefix}condition") or "")])
        if f"{prefix}group" in result:
            commands.extend(["group", str(result.get(f"{prefix}group") or "")])
        if f"{prefix}session" in result:
            commands.extend(["session", parse_optional_int_text(result.get(f"{prefix}session"))])
        if f"{prefix}run" in result:
            commands.extend(["run", parse_optional_int_text(result.get(f"{prefix}run"))])
    return commands


def _display_path(info: dict[str, Any]) -> str:
    filename = str(info.get("filename") or "")
    filepath = str(info.get("filepath") or "")
    return f"{filepath}/{filename}" if filepath and filename else filename


def _display_optional_number(value: Any) -> str:
    if value in (None, "", []):
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


__all__ = ["pop_study", "pop_study_dialog_spec"]
