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
    commands: Any = None,
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
    commands = options.pop("commands", commands)
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
        design=design,
        commands=commands,
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
    datasetinfo = checked.get("datasetinfo") or []
    visible_rows = max(10, len(datasetinfo))
    controls: list[ControlSpec] = [
        ControlSpec("text", "Create a new STUDY set", font_weight="bold"),
        ControlSpec("spacer"),
        ControlSpec("text", "STUDY set name:"),
        ControlSpec("edit", tag="name", value=str(checked.get("name") or "")),
        ControlSpec("spacer"),
        ControlSpec("text", "STUDY set task name:"),
        ControlSpec("edit", tag="task", value=str(checked.get("task") or "")),
        ControlSpec("spacer"),
        ControlSpec("text", "STUDY set notes:"),
        ControlSpec("edit", tag="notes", value=str(checked.get("notes") or "")),
        ControlSpec("spacer"),
        ControlSpec("spacer"),
        ControlSpec("text", "dataset filename", font_weight="bold"),
        ControlSpec("text", "browse", font_weight="bold"),
        ControlSpec("text", "subject", font_weight="bold"),
        ControlSpec("text", "session", font_weight="bold"),
        ControlSpec("text", "run", font_weight="bold"),
        ControlSpec("text", "condition", font_weight="bold"),
        ControlSpec("text", "group", font_weight="bold"),
        ControlSpec("pushbutton", "Select by r.v.", tag="select_by_rv", enabled=False),
        ControlSpec("spacer"),
    ]
    for index in range(1, visible_rows + 1):
        info = datasetinfo[index - 1] if index <= len(datasetinfo) else {}
        has_dataset = bool(info)
        controls.extend(
            (
                ControlSpec("text", str(index)),
                ControlSpec(
                    "edit",
                    tag=f"dataset_{index}_filename",
                    value=_display_path(info),
                    enabled=False,
                ),
                ControlSpec("pushbutton", "...", tag=f"dataset_{index}_browse", enabled=False),
                ControlSpec(
                    "edit",
                    tag=f"dataset_{index}_subject",
                    value=str(info.get("subject") or ""),
                    enabled=has_dataset,
                ),
                ControlSpec(
                    "edit",
                    tag=f"dataset_{index}_session",
                    value=_display_optional_number(info.get("session")),
                    enabled=has_dataset,
                ),
                ControlSpec(
                    "edit",
                    tag=f"dataset_{index}_run",
                    value=_display_optional_number(info.get("run")),
                    enabled=has_dataset,
                ),
                ControlSpec(
                    "edit",
                    tag=f"dataset_{index}_condition",
                    value=str(info.get("condition") or ""),
                    enabled=has_dataset,
                ),
                ControlSpec(
                    "edit",
                    tag=f"dataset_{index}_group",
                    value=str(info.get("group") or ""),
                    enabled=has_dataset,
                ),
                ControlSpec("pushbutton", "All comp.", tag=f"dataset_{index}_components", enabled=False),
                ControlSpec("pushbutton", "Clear", tag=f"dataset_{index}_clear", enabled=False),
            )
        )
    controls.extend(
        (
            ControlSpec(
                "text",
                "Important note: Removed datasets will not be saved before being deleted from EEGPrep memory",
            ),
            ControlSpec("spacer"),
            ControlSpec("pushbutton", "<", tag="previous_page", enabled=False),
            ControlSpec("text", "Page 1"),
            ControlSpec("pushbutton", ">", tag="next_page", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "", tag="delete_cluster_info", value=False, enabled=False),
            ControlSpec(
                "text",
                "Delete cluster information (to allow loading new datasets, set new components for clustering, etc.)",
                enabled=False,
            ),
        )
    )
    header_geometry = (0.2, 1.05, 0.35, 0.4, 0.35, 0.25, 0.6, 0.4, 0.6, 0.3)
    geometry = [(1,), (0.2, 1, 3.5), (0.2, 1, 3.5), (0.2, 1, 3.5), (1,), header_geometry]
    geometry.extend(header_geometry for _index in range(visible_rows))
    geometry.extend(((1,), (1, 0.2, 0.3, 0.2, 1), (1,), (0.14, 3)))
    return DialogSpec(
        title=title,
        controls=tuple(controls),
        geometry=tuple(geometry),
        function_name="pop_study",
        eeglab_source="functions/studyfunc/pop_study.m",
        help_text="pop_study",
        size=(1080, 834),
        scrollable=visible_rows > 10,
        content_margins=(26, 20, 26, 16),
        row_spacing=8,
        button_size=(54, 20),
        extra_stylesheet="""
            QDialog#pop_study QLabel,
            QDialog#pop_study QCheckBox,
            QDialog#pop_study QPushButton,
            QDialog#pop_study QLineEdit {
                font-size: 15px;
            }
            QDialog#pop_study QLineEdit {
                min-height: 28px;
                max-height: 28px;
                padding: 0 2px;
            }
            QDialog#pop_study QPushButton {
                min-height: 27px;
                max-height: 27px;
                padding: 0 4px;
            }
            QDialog#pop_study QPushButton#select_by_rv {
                min-width: 76px;
                max-width: 76px;
            }
            QDialog#pop_study QPushButton#previous_page,
            QDialog#pop_study QPushButton#next_page {
                min-width: 46px;
                max-width: 46px;
            }
        """,
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
