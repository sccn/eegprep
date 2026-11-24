"""Create a STUDY by browsing for dataset files."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.studyfunc._study_utils import MatlabLiteral, build_matlab_call, matlab_command_cell
from eegprep.functions.studyfunc.std_editset import std_editset


def pop_studywizard(
    filenames: list[str] | tuple[str, ...] | None = None,
    *,
    name: str = "EEGPrep study",
    gui: bool = False,
    renderer: Any = None,
    return_com: bool = False,
) -> Any:
    """Load selected datasets and create a STUDY.

    ``gui=True`` shows one row per file (subject, session, run, condition,
    group, task) and builds the STUDY only after Ok.
    """
    selected = [str(filename) for filename in filenames or []]
    metadata: list[dict[str, str]] = []
    if gui or renderer is not None:
        result = inputgui(pop_studywizard_dialog_spec(selected, name), renderer=renderer)
        if not result:
            return (None, [], "") if return_com else (None, [])
        name = str(result.get("name") or name).strip() or name
        selected = []
        for index in range(1, _row_count(result) + 1):
            filename = str(result.get(f"file{index}") or "").strip()
            if not filename:
                continue
            selected.append(filename)
            metadata.append(
                {
                    key: str(result.get(f"{key}{index}") or "").strip()
                    for key in ("subject", "session", "run", "condition", "group", "task")
                }
            )
    if not selected:
        raise ValueError(
            "pop_studywizard: no dataset files were selected. Choose one or more .set files, then press Ok."
        )
    commands = []
    for index, (filename, fields) in enumerate(zip(selected, metadata or [{}] * len(selected)), start=1):
        command = {"index": index, "load": filename, "subject": fields.get("subject") or f"S{index}"}
        for key in ("session", "run", "condition", "group", "task"):
            if fields.get(key):
                command[key] = fields[key]
        commands.append(command)
    study, alleeg, _command = std_editset(None, [], name=name, commands=commands, updatedat="off", return_com=True)
    command = build_matlab_call(
        ("STUDY", "ALLEEG"),
        "std_editset",
        "STUDY",
        MatlabLiteral("[]"),
        commands=commands,
        updatedat="on",
        dict_formatter=matlab_command_cell,
    )
    command += "\n" + build_matlab_call(("STUDY", "ALLEEG"), "std_checkset", "STUDY", "ALLEEG")
    return (study, alleeg, command) if return_com else (study, alleeg)


_ROW_GEOMETRY = (0.2, 1.05, 0.35, 0.4, 0.35, 0.25, 0.6, 0.4, 0.6, 0.3)
_ROW_HEADERS = (
    "",
    "Raw dataset filename",
    "browse",
    "subject",
    "session",
    "run",
    "condition",
    "group",
    "task",
    "",
)


def pop_studywizard_dialog_spec(filenames: list[str], name: str) -> DialogSpec:
    """Return the browse-for-datasets grid from ``pop_studywizard.m``.

    ``supergui`` displays the source button ``CLear`` as ``Clear``. The header
    row names ``browse`` and leaves the Clear column unlabeled.
    """
    controls = [
        ControlSpec("text", "Create a STUDY set", font_weight="bold"),
        ControlSpec("text", "STUDY name"),
        ControlSpec("edit", tag="name", value=name),
    ]
    geometry: list[tuple[float, ...] | tuple[int, ...]] = [(1,), (1, 3)]
    controls.extend(ControlSpec("text", header) for header in _ROW_HEADERS)
    geometry.append(_ROW_GEOMETRY)
    for index, filename in enumerate(filenames, start=1):
        controls.extend(
            [
                ControlSpec("text", str(index)),
                ControlSpec("edit", tag=f"file{index}", value=filename),
                ControlSpec(
                    "pushbutton",
                    "...",
                    tag=f"browse{index}",
                    callback=CallbackSpec(
                        "select_file",
                        {
                            "button": f"browse{index}",
                            "target": f"file{index}",
                            "caption": "Choose dataset to add -- pop_studywizard()",
                        },
                    ),
                ),
                ControlSpec("edit", tag=f"subject{index}", value=f"S{index}"),
                ControlSpec("edit", tag=f"session{index}"),
                ControlSpec("edit", tag=f"run{index}"),
                ControlSpec("edit", tag=f"condition{index}"),
                ControlSpec("edit", tag=f"group{index}"),
                ControlSpec("edit", tag=f"task{index}"),
                ControlSpec(
                    "pushbutton",
                    "Clear",
                    tag=f"clear{index}",
                    callback=CallbackSpec(
                        "clear_widgets",
                        {
                            "button": f"clear{index}",
                            "targets": [
                                f"file{index}",
                                f"subject{index}",
                                f"session{index}",
                                f"run{index}",
                                f"condition{index}",
                                f"group{index}",
                                f"task{index}",
                            ],
                        },
                    ),
                ),
            ]
        )
        geometry.append(_ROW_GEOMETRY)
    return DialogSpec(
        title="Create STUDY set -- pop_studywizard()",
        function_name="pop_studywizard",
        help_text="pop_studywizard",
        eeglab_source="functions/studyfunc/pop_studywizard.m",
        geometry=tuple(geometry),
        controls=tuple(controls),
        size=(1186, 500),
        scrollable=False,
    )


def _row_count(result: dict[str, Any]) -> int:
    indexes = []
    for key in result:
        if key.startswith("file") and key[4:].isdigit():
            indexes.append(int(key[4:]))
    return max(indexes, default=0)
