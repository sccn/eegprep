"""BIDS metadata editors and a file-level dataset check."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.EEG_BIDS.bids_list_eeg_files import bids_list_eeg_files
from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids

_TASK_FIELDS = (
    ("Name", "* Dataset name"),
    ("TaskName", "* Task name (no space)"),
    ("README", "README"),
    ("TaskDescription", "Participant task description"),
    ("Instructions", "Participant instructions"),
    ("Authors", "Authors"),
    ("ReferencesAndLinks", "References and links"),
    ("CogAtlasID", "Task-relevant Cognitive Atlas term"),
    ("CogPOID", "Task-relevant CogPO term"),
    ("InstitutionName", "Institution"),
    ("InstitutionalDepartmentName", "Department"),
    ("InstitutionAddress", "Institution location"),
    ("CapManufacturer", "Cap manufacturer"),
    ("CapManufacturersModelName", "Cap model"),
    ("EEGReference", "EEG reference location"),
    ("EEGGround", "EEG ground electrode location"),
    ("EEGPlacementScheme", "EEG montage system (10-20, 10-10, custom)"),
    ("Manufacturer", "EEG amplifier maker"),
    ("ManufacturersModelName", "EEG amplifier model"),
    ("DeviceSerialNumber", "EEG amplifier serial #"),
    ("SoftwareVersions", "EEG acquisition software version"),
    ("HardwareFilters", "Hardware filters"),
    ("SoftwareFilters", "Software filters"),
)
_PARTICIPANT_FIELDS = (
    ("participant_id", "Subject code"),
    ("Gender", "Gender"),
    ("Age", "Age"),
    ("Group", "Group"),
    ("HeadCircumference", "Head circumference"),
    ("SubjectArtefactDescription", "Subject artefact description"),
)
_EVENT_FIELDS = (
    ("onset", "onset"),
    ("duration", "duration"),
    ("sample", "sample"),
    ("value", "value"),
    ("trial_type", "trial_type"),
    ("stim_file", "stim_file"),
    ("response_time", "response_time"),
)


def bids_exporter(
    EEG: dict[str, Any] | list[dict[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    *,
    entries: list[dict[str, Any]] | None = None,
    gui: bool = False,
    renderer: Any = None,
) -> tuple[str, str]:
    """Export datasets, or raw files chosen from the welcome page and file grid.

    With no loaded dataset, ``gui=True`` opens the welcome page and then a
    file grid. Cancel returns empty strings and writes nothing.
    """
    if entries is not None or (EEG is None and (gui or renderer is not None)):
        return _export_wizard(output_dir, entries, renderer)
    if EEG is None:
        raise ValueError(
            "bids_exporter: no datasets were given. "
            "Use File > BIDS tools > BIDS export wizard and choose the raw files to export."
        )
    return pop_exportbids(EEG, output_dir, return_com=True)


def pop_taskinfo(
    target: dict[str, Any],
    *,
    gui: bool = False,
    renderer: Any = None,
    datasets: list[dict[str, Any]] | None = None,
    **metadata: Any,
) -> tuple[dict[str, Any], str]:
    """Attach BIDS task metadata to an EEG or STUDY dict.

    ``gui=True`` opens the field dialog from ``pop_taskinfo.m``. Keyword
    arguments still update ``etc.bids.task`` without a dialog.
    """
    if gui or renderer is not None:
        collected = _dialog_metadata(pop_taskinfo_dialog_spec(), renderer, _task_initial(target))
        if collected is None:
            return target, ""
        metadata = {**metadata, **collected}
    updated, command = _update_bids_metadata(target, "task", metadata, "pop_taskinfo")
    _fan_out(datasets, "task", metadata)
    return updated, command


def pop_participantinfo(
    target: dict[str, Any],
    study: dict[str, Any] | str | None = None,
    *args: Any,
    gui: bool = False,
    renderer: Any = None,
    datasets: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
    column_info: dict[str, dict[str, str]] | None = None,
    **metadata: Any,
) -> tuple[dict[str, Any], str]:
    """Attach BIDS participant metadata to an EEG or STUDY dict.

    EEGLAB's second argument is ``STUDY``. Pass that dict positionally when
    replaying a history line. ``rows`` is one dict per dataset
    (participant_id, Gender, Age, Group, and the other participants.tsv
    columns). ``column_info`` is the column Description/Units table written
    to participants.json.
    """
    if isinstance(study, str):
        args = (study, *args)
    options = parse_key_value_args(args, metadata)
    rows = options.pop("rows", rows)
    column_info = options.pop("column_info", column_info)
    metadata = options
    if rows is not None or (datasets and gui and renderer is None):
        if rows is None:
            edited, dialog_columns = _participant_table(datasets or [])
            if edited is None:
                return target, ""
            column_info = dialog_columns or column_info
        else:
            edited = rows
        _store_participant_rows(datasets or [target], edited, column_info)
        return target, _participant_history()
    if gui or renderer is not None:
        collected = _dialog_metadata(pop_participantinfo_dialog_spec(), renderer, _participant_initial(target))
        if collected is None:
            return target, ""
        if not str(collected.get("participant_id") or "").strip():
            raise ValueError(
                "pop_participantinfo: No subject ID found. Fill in Subject code (participant_id), then press Ok."
            )
        metadata = {**metadata, **collected}
    updated, command = _update_bids_metadata(target, "participant", metadata, "pop_participantinfo")
    _fan_out(datasets, "participant", metadata)
    return updated, command


def pop_eventinfo(
    target: dict[str, Any],
    *,
    gui: bool = False,
    renderer: Any = None,
    datasets: list[dict[str, Any]] | None = None,
    event_fields: list[dict[str, Any]] | None = None,
    **metadata: Any,
) -> tuple[dict[str, Any], str]:
    """Attach BIDS event metadata to an EEG or STUDY dict.

    ``event_fields`` is the event table: each row has ``bids_field`` plus
    EEGLAB Field, Levels, LongName, Description, and Units. Export writes
    those columns into ``*_events.json``.
    """
    if event_fields is not None or (gui and renderer is None and datasets is not None):
        edited = event_fields if event_fields is not None else _event_table(_section(target, "event"))
        if edited is None:
            return target, ""
        metadata = {**metadata, **_event_field_map(edited)}
    elif gui or renderer is not None:
        collected = _dialog_metadata(pop_eventinfo_dialog_spec(), renderer, _section(target, "event"))
        if collected is None:
            return target, ""
        metadata = {**metadata, **collected}
    updated, command = _update_bids_metadata(target, "event", metadata, "pop_eventinfo")
    _fan_out(datasets, "event", metadata)
    return updated, command


def pop_taskinfo_dialog_spec() -> DialogSpec:
    """Return the task and EEG acquisition fields from ``pop_taskinfo.m``."""
    controls = [ControlSpec("text", "BIDS task information", font_weight="bold")]
    for tag, label in _TASK_FIELDS:
        controls.extend((ControlSpec("text", label), ControlSpec("edit", tag=tag)))
    controls.extend(
        (
            ControlSpec("text", "Line frequency (Hz)"),
            ControlSpec("popupmenu", "Select|50|60", tag="PowerLineFrequency", value=1),
        )
    )
    return DialogSpec(
        title="BIDS task information -- pop_taskinfo()",
        function_name="pop_taskinfo",
        eeglab_source="plugins/EEG-BIDS/pop_taskinfo.m",
        geometry=tuple([1] + [[1, 2]] * (len(_TASK_FIELDS) + 1)),
        controls=tuple(controls),
        scrollable=True,
    )


def pop_participantinfo_dialog_spec() -> DialogSpec:
    """Return the participant columns from ``pop_participantinfo.m`` line 160."""
    controls = [ControlSpec("text", "Participant information", font_weight="bold")]
    for tag, label in _PARTICIPANT_FIELDS:
        controls.extend((ControlSpec("text", label), ControlSpec("edit", tag=tag)))
    return DialogSpec(
        title="Edit BIDS participant info - pop_participantinfo",
        function_name="pop_participantinfo",
        eeglab_source="plugins/EEG-BIDS/pop_participantinfo.m",
        geometry=tuple([1] + [[1, 2]] * len(_PARTICIPANT_FIELDS)),
        controls=tuple(controls),
    )


def pop_eventinfo_dialog_spec() -> DialogSpec:
    """Return the event fields from ``pop_eventinfo.m`` line 59."""
    controls = [ControlSpec("text", "BIDS information for EEG.event fields", font_weight="bold")]
    for tag, label in _EVENT_FIELDS:
        controls.extend((ControlSpec("text", label), ControlSpec("edit", tag=tag)))
    return DialogSpec(
        title="Edit BIDS event info - pop_eventinfo",
        function_name="pop_eventinfo",
        eeglab_source="plugins/EEG-BIDS/pop_eventinfo.m",
        geometry=tuple([1] + [[1, 2]] * len(_EVENT_FIELDS)),
        controls=tuple(controls),
    )


_VALIDATOR_NOTE = (
    "validate_bids: this was a sidecar check only. The official bids-validator was not found on PATH. "
    "Install it with `npm install -g bids-validator` or `deno install -A jsr:@bids/validator`, "
    "then run validate_bids again."
)


def validate_bids(root: str | Path) -> dict[str, list[str] | str | None]:
    """Check BIDS sidecars, then the official bids-validator when it is installed.

    ``pop_validatebids.m`` runs the bids-validator plugin. This function runs
    ``bids-validator`` or ``npx --no-install bids-validator`` when that command
    is on PATH, and ``deno run -A jsr:@bids/validator`` when ``deno`` is on PATH.
    It does not write a report file into the dataset. ``note`` tells the GUI
    when only the sidecar check ran.
    """
    path = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        errors.append(
            f"validate_bids: BIDS path does not exist: {path}. Pass the folder that contains dataset_description.json."
        )
        return {"errors": errors, "warnings": warnings, "note": _VALIDATOR_NOTE}
    if not (path / "dataset_description.json").exists():
        errors.append(
            f"validate_bids: dataset_description.json is missing from {path}. "
            "Export the folder with pop_exportbids, which writes that file."
        )
    if not (path / "README").exists():
        errors.append(
            f"validate_bids: README is missing from {path}. "
            "Export the folder with pop_exportbids, which writes README, *_eeg.json, and *_electrodes.tsv."
        )
    try:
        files = bids_list_eeg_files(str(path))
    except Exception as exc:
        errors.append(f"validate_bids: could not list EEG files in {path}. {exc}")
        files = []
    if not files:
        errors.append(
            f"validate_bids: no supported EEG files were found in {path}. "
            "Export a .set, .edf, .bdf, or BrainVision file with pop_exportbids."
        )
    eeg_json = list(path.rglob("*_eeg.json"))
    electrodes = list(path.rglob("*_electrodes.tsv"))
    if files and not eeg_json:
        errors.append(
            f"validate_bids: no *_eeg.json sidecar in {path}. "
            "Re-export with pop_exportbids so each recording has an eeg.json file."
        )
    if files and not electrodes:
        errors.append(
            f"validate_bids: no *_electrodes.tsv file in {path}. "
            "Re-export with pop_exportbids so each recording has an electrodes.tsv file."
        )
    _check_coordsystem_names(path, errors)
    coordsystems = list(path.rglob("*_coordsystem.json")) + list(path.glob("coordsystem.json"))
    if electrodes and not coordsystems:
        errors.append(
            f"validate_bids: electrodes.tsv is present in {path} but no *_coordsystem.json was found. "
            "Re-export with pop_exportbids, which writes the coordinate-system sidecar beside electrodes.tsv."
        )
    note = _run_official_validator(path, errors, warnings)
    return {"errors": errors, "warnings": warnings, "note": note}


def _check_coordsystem_names(path: Path, errors: list[str]) -> None:
    """Flag coordinate-system names that BIDS rejects, even without bids-validator."""
    forbidden = ("_run-", "_task-", "_desc-", "_rec-")
    for file in path.rglob("*coordsystem.json"):
        if any(token in file.name for token in forbidden):
            errors.append(
                f"validate_bids: {file.name} uses a BIDS entity that coordsystem.json does not allow. "
                "Remove run-, task-, desc-, and rec- from that filename, then validate again."
            )


_PARTICIPANT_COLUMNS = (
    "participant_id",
    "Gender",
    "Age",
    "Group",
    "HeadCircumference",
    "SubjectArtefactDescription",
)
_EVENT_COLUMNS = ("bids_field", "eeglab_field", "Levels", "LongName", "Description", "Units")
_WIZARD_PAGE_ROWS = 10
_WIZARD_FIELDS = ("file", "subject", "session", "run", "condition", "group", "task")
_WIZARD_HEADERS = (
    "Raw dataset filename",
    "browse",
    "subject",
    "session",
    "run",
    "condition",
    "group",
    "task",
    "CLear",
)


def _export_wizard(
    output_dir: str | Path | None,
    entries: list[dict[str, Any]] | None,
    renderer: Any,
) -> tuple[str, str]:
    if entries is None:
        welcome = inputgui(bids_exporter_welcome_spec(), renderer=renderer)
        if welcome is None:
            return "", ""
        if renderer is not None and hasattr(renderer, "run"):
            grid = inputgui(bids_exporter_grid_spec(), renderer=renderer)
            if grid is None:
                return "", ""
            output_dir = str(grid.get("output") or output_dir or "").strip()
            entries = _entries_from_grid(grid)
        else:
            collected = _qt_export_grid()
            if collected is None:
                return "", ""
            output_dir, entries = collected
    if not entries:
        raise ValueError(
            "bids_exporter: no files were entered. Fill at least one raw-file row on the export grid, then press Ok."
        )
    if not output_dir:
        raise ValueError(
            "bids_exporter: an output folder is required. Enter the output folder on the export grid, then press Ok."
        )
    datasets = []
    for entry in entries:
        dataset = _load_recording(str(entry["file"]))
        if entry.get("subject"):
            dataset["subject"] = entry["subject"]
        for key in ("session", "run", "condition", "group", "task"):
            if entry.get(key):
                dataset[key] = entry[key]
        datasets.append(dataset)
    task = str(entries[0].get("task") or "eeg")
    return pop_exportbids(datasets, output_dir, task=task, return_com=True)


def bids_exporter_welcome_spec() -> DialogSpec:
    """Return the BIDS exporter welcome page from ``bids_exporter.m``."""
    return DialogSpec(
        title="BIDS export wizard -- bids_exporter()",
        function_name="bids_exporter",
        eeglab_source="plugins/EEG-BIDS/bids_exporter.m",
        geometry=((1,), (1,)),
        controls=(
            ControlSpec("text", "Welcome to the EEG-BIDS exporter tool!", font_weight="bold"),
            ControlSpec(
                "text",
                "Continue to choose raw files, subject, session, run, condition, group, and task. "
                "No dataset needs to be loaded.",
            ),
        ),
    )


def bids_exporter_grid_spec(row_count: int | None = None) -> DialogSpec:
    """Return one page of the raw-file grid from ``pop_studywizard.m``.

    The interactive wizard shows ``_WIZARD_PAGE_ROWS`` rows and pages with
    ``<`` and ``>`` for any number of datasets. ``row_count`` selects how many
    rows this spec lists when a caller builds the page itself.
    """
    count = _WIZARD_PAGE_ROWS if row_count is None else int(row_count)
    if count < 1:
        raise ValueError(
            "bids_exporter: the file grid needs at least one row. "
            "Pass row_count as a positive whole number, then open the wizard again."
        )
    controls: list[ControlSpec] = [
        ControlSpec("text", "Output folder"),
        ControlSpec("edit", tag="output"),
        ControlSpec("text", "Select datasets", font_weight="bold"),
    ]
    geometry: list[tuple[float, ...] | tuple[int, ...]] = [(1, 3), (1,)]
    controls.extend(ControlSpec("text", header) for header in _WIZARD_HEADERS)
    row_geometry = (3, 0.45, 1, 1, 1, 1, 1, 1, 0.6)
    geometry.append(row_geometry)
    for index in range(1, count + 1):
        controls.extend(
            [
                ControlSpec("edit", tag=f"file{index}"),
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
                ControlSpec("edit", tag=f"subject{index}"),
                ControlSpec("edit", tag=f"session{index}"),
                ControlSpec("edit", tag=f"run{index}"),
                ControlSpec("edit", tag=f"condition{index}"),
                ControlSpec("edit", tag=f"group{index}"),
                ControlSpec("edit", tag=f"task{index}"),
                ControlSpec(
                    "pushbutton",
                    "CLear",
                    tag=f"clear{index}",
                    callback=CallbackSpec(
                        "clear_widgets",
                        {"targets": [f"{field}{index}" for field in _WIZARD_FIELDS]},
                    ),
                ),
            ]
        )
        geometry.append(row_geometry)
    controls.extend(
        [
            ControlSpec("pushbutton", "<", tag="prev_page"),
            ControlSpec("text", "Page 1", tag="page"),
            ControlSpec("pushbutton", ">", tag="next_page"),
        ]
    )
    geometry.append((1, 0.3, 1))
    return DialogSpec(
        title="BIDS export wizard -- bids_exporter()",
        function_name="bids_exporter",
        eeglab_source="functions/studyfunc/pop_studywizard.m",
        geometry=tuple(geometry),
        controls=tuple(controls),
        scrollable=False,
    )


def _entries_from_grid(grid: dict[str, Any]) -> list[dict[str, str]]:
    entries = []
    for index in _wizard_row_indexes(grid):
        filename = str(grid.get(f"file{index}") or "").strip()
        if not filename:
            continue
        entries.append(
            {
                "file": filename,
                "subject": str(grid.get(f"subject{index}") or "").strip(),
                "session": str(grid.get(f"session{index}") or "").strip(),
                "run": str(grid.get(f"run{index}") or "").strip(),
                "condition": str(grid.get(f"condition{index}") or "").strip(),
                "group": str(grid.get(f"group{index}") or "").strip(),
                "task": str(grid.get(f"task{index}") or "").strip(),
            }
        )
    return entries


class _ExportGrid:
    def __init__(
        self,
        dialog: Any,
        state: dict[str, Any],
        edits: list[dict[str, Any]],
        output: Any,
        page_label: Any,
        save_page: Any,
        shift: Any,
    ) -> None:
        self.dialog = dialog
        self.state = state
        self.edits = edits
        self.output = output
        self.page_label = page_label
        self.save_page = save_page
        self.shift = shift

    def exec(self) -> int:
        return int(self.dialog.exec())

    def show(self) -> None:
        self.dialog.show()

    def close(self) -> None:
        self.dialog.close()

    def findChildren(self, kind: Any) -> list[Any]:
        return list(self.dialog.findChildren(kind))


def _qt_export_grid() -> tuple[str, list[dict[str, str]]] | None:
    """Show the paged export grid and return the output folder plus filled rows."""
    from PySide6.QtWidgets import QDialog

    grid = _export_grid_dialog()
    if grid.exec() != int(QDialog.DialogCode.Accepted):
        return None
    grid.save_page()
    entries = [dict(row) for row in grid.state["rows"] if str(row.get("file") or "").strip()]
    return str(grid.output.text()).strip(), entries


def _export_grid_dialog() -> _ExportGrid:
    """Build the 10-row ``pop_studywizard`` page without showing it."""
    try:
        from PySide6.QtWidgets import (
            QDialog,
            QFileDialog,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QVBoxLayout,
        )
    except ImportError as exc:
        raise ImportError(
            "bids_exporter: the export grid needs PySide6. "
            "Install it with `pip install eegprep[gui]`, then open the wizard again."
        ) from exc

    dialog = QDialog()
    dialog.setWindowTitle("BIDS export wizard -- bids_exporter()")
    state: dict[str, Any] = {
        "page": 0,
        "rows": [{field: "" for field in _WIZARD_FIELDS} for _ in range(_WIZARD_PAGE_ROWS)],
    }
    edits: list[dict[str, Any]] = []
    grid = QGridLayout()
    for column, header in enumerate(_WIZARD_HEADERS):
        label = QLabel("" if header == "browse" else header)
        label.setObjectName(f"header-{header}")
        grid.addWidget(label, 0, column)
    for row in range(1, _WIZARD_PAGE_ROWS + 1):
        widgets: dict[str, Any] = {}
        file_edit = QLineEdit()
        file_edit.setObjectName(f"file{row}")
        widgets["file"] = file_edit
        grid.addWidget(file_edit, row, 0)
        browse = QPushButton("...")
        browse.setObjectName(f"browse{row}")

        def choose(edit: Any = file_edit) -> None:
            name, _selected = QFileDialog.getOpenFileName(dialog, "Choose dataset to add -- pop_studywizard()")
            if name:
                edit.setText(name)

        browse.clicked.connect(choose)
        grid.addWidget(browse, row, 1)
        for column, field in enumerate(_WIZARD_FIELDS[1:], start=2):
            edit = QLineEdit()
            edit.setObjectName(f"{field}{row}")
            widgets[field] = edit
            grid.addWidget(edit, row, column)
        clear = QPushButton("CLear")
        clear.setObjectName(f"clear{row}")

        def wipe(_checked: bool = False, row_widgets: dict[str, Any] = widgets) -> None:
            for edit in row_widgets.values():
                edit.setText("")

        clear.clicked.connect(wipe)
        grid.addWidget(clear, row, len(_WIZARD_FIELDS) + 1)
        edits.append(widgets)

    page_label = QLabel("Page 1")
    page_label.setObjectName("page")
    previous = QPushButton("<")
    previous.setObjectName("prev_page")
    following = QPushButton(">")
    following.setObjectName("next_page")
    output = QLineEdit()
    output.setObjectName("output")

    def blank() -> dict[str, str]:
        return {field: "" for field in _WIZARD_FIELDS}

    def ensure(page: int) -> None:
        needed = (page + 1) * _WIZARD_PAGE_ROWS
        while len(state["rows"]) < needed:
            state["rows"].append(blank())

    def save_page() -> None:
        start = state["page"] * _WIZARD_PAGE_ROWS
        ensure(state["page"])
        for offset, widgets in enumerate(edits):
            for field, widget in widgets.items():
                state["rows"][start + offset][field] = widget.text().strip()

    def load_page() -> None:
        ensure(state["page"])
        start = state["page"] * _WIZARD_PAGE_ROWS
        for offset, widgets in enumerate(edits):
            for field, widget in widgets.items():
                widget.setText(state["rows"][start + offset][field])
        page_label.setText(f"Page {state['page'] + 1}")

    def shift(delta: int) -> None:
        save_page()
        state["page"] = max(0, state["page"] + delta)
        load_page()

    previous.clicked.connect(lambda: shift(-1))
    following.clicked.connect(lambda: shift(1))
    help_button = QPushButton("Help")
    help_button.setObjectName("help")
    cancel_button = QPushButton("Cancel")
    cancel_button.setObjectName("cancel")
    ok_button = QPushButton("Ok")
    ok_button.setObjectName("ok")
    help_button.clicked.connect(lambda: _show_export_help(dialog))
    cancel_button.clicked.connect(dialog.reject)
    ok_button.clicked.connect(dialog.accept)
    buttons = QHBoxLayout()
    buttons.addWidget(help_button)
    buttons.addStretch(1)
    buttons.addWidget(cancel_button)
    buttons.addWidget(ok_button)
    pages = QHBoxLayout()
    pages.addWidget(previous)
    pages.addWidget(page_label)
    pages.addWidget(following)
    layout = QVBoxLayout(dialog)
    output_row = QHBoxLayout()
    output_row.addWidget(QLabel("Output folder"))
    output_row.addWidget(output)
    layout.addLayout(output_row)
    layout.addLayout(grid)
    layout.addLayout(pages)
    layout.addLayout(buttons)
    dialog.adjustSize()
    return _ExportGrid(dialog, state, edits, output, page_label, save_page, shift)


def _show_export_help(parent: Any) -> None:
    from eegprep.functions.guifunc.pophelp import pophelp

    pophelp("pop_exportbids", parent=parent)


def _wizard_row_indexes(grid: dict[str, Any]) -> list[int]:
    indexes = [int(key[4:]) for key in grid if key.startswith("file") and key[4:].isdigit()]
    return sorted(indexes)


def _participant_history() -> str:
    """Return the history line EEGLAB records for this dialog.

    EEGLAB's ``pop_participantinfo`` is interactive-only. EEGPrep applies
    ``rows`` in Python, and the recorded command stays the line EEGLAB writes.
    Replaying ``'rows'`` would open the table and never apply it.
    """
    return "[EEG, STUDY, command] = pop_participantinfo(EEG, STUDY);"


def _participant_table(datasets: list[dict[str, Any]]) -> tuple[list[dict[str, str]] | None, dict[str, dict[str, str]]]:
    rows = []
    for dataset in datasets:
        stored = _section(dataset, "participant")
        row = {column: str(stored.get(column) or "") for column in _PARTICIPANT_COLUMNS}
        if not row["participant_id"]:
            row["participant_id"] = _subject_text(dataset.get("subject"))
        rows.append(row)
    defaults = {
        "Gender": "Participant gender",
        "Age": "Participant age",
        "Group": "Participant group",
        "HeadCircumference": "Head circumference",
        "SubjectArtefactDescription": "Subject artefact description",
    }
    column_rows = [{"Field": field, "Description": text, "Units": ""} for field, text in defaults.items()]
    QDialog, QDialogButtonBox, _table, _item, QVBoxLayout = _qt_widgets("pop_participantinfo")
    dialog = QDialog()
    dialog.setWindowTitle("Edit BIDS participant info - pop_participantinfo")
    participants = _fill_table(dialog, "bids-table", _PARTICIPANT_COLUMNS, rows)
    columns = _fill_table(dialog, "bids-column-table", ("Field", "Description", "Units"), column_rows)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout = QVBoxLayout(dialog)
    layout.addWidget(participants)
    layout.addWidget(columns)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None, {}
    return _read_table(participants, _PARTICIPANT_COLUMNS), {
        item["Field"]: {key: item[key] for key in ("Description", "Units") if item.get(key)}
        for item in _read_table(columns, ("Field", "Description", "Units"))
        if item.get("Field")
    }


def _event_table(stored: dict[str, Any]) -> list[dict[str, str]] | None:
    return _qt_table("Edit BIDS event info - pop_eventinfo", _EVENT_COLUMNS, _event_rows(stored))


def _event_rows(stored: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    known = {field for field, _label in _EVENT_FIELDS}
    fields = list(_EVENT_FIELDS) + [
        (name, name) for name in stored if name not in known and isinstance(stored[name], dict)
    ]
    for bids_field, eeglab_field in fields:
        current = stored.get(bids_field)
        if not isinstance(current, dict):
            current = {"Description": "" if current is None else str(current), "eeglab_field": eeglab_field}
        rows.append(
            {
                "bids_field": bids_field,
                "eeglab_field": str(current.get("eeglab_field") or eeglab_field),
                "Levels": _levels_text(current.get("Levels")),
                "LongName": str(current.get("LongName") or ""),
                "Description": str(current.get("Description") or ""),
                "Units": str(current.get("Units") or ""),
            }
        )
    return rows


def _levels_text(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}={item}" for key, item in value.items())
    return "" if value in (None, "") else str(value)


def _levels_value(text: str) -> str | dict[str, str]:
    if "=" not in text:
        return text
    levels = {}
    for part in text.split(","):
        if "=" not in part:
            continue
        key, item = part.split("=", 1)
        key, item = key.strip(), item.strip()
        if key:
            levels[key] = item
    return levels or text


def _load_recording(path: str) -> dict[str, Any]:
    suffix = Path(path).suffix.lower()
    if suffix in {".edf", ".bdf", ".gdf"}:
        from eegprep.functions.popfunc.pop_biosig import pop_biosig

        loaded = pop_biosig(path)
    elif suffix == ".vhdr":
        from eegprep.functions.popfunc.pop_loadbv import pop_loadbv

        loaded = pop_loadbv(path)
    elif suffix == ".set":
        loaded = pop_loadset(path)
    else:
        from eegprep.functions.popfunc.pop_fileio import pop_fileio

        loaded = pop_fileio(path)
    if isinstance(loaded, tuple):
        loaded = loaded[0]
    if not isinstance(loaded, dict):
        raise ValueError(
            f"bids_exporter: {path} did not load as an EEG dataset. "
            "Use a .set, .edf, .bdf, .gdf, or BrainVision .vhdr file, then export again."
        )
    return loaded


def _qt_table(title: str, columns: tuple[str, ...], rows: list[dict[str, str]]) -> list[dict[str, str]] | None:
    QDialog, QDialogButtonBox, _table, _item, QVBoxLayout = _qt_widgets("pop_eventinfo")
    dialog = QDialog()
    dialog.setWindowTitle(title)
    table = _fill_table(dialog, "bids-table", columns, rows)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout = QVBoxLayout(dialog)
    layout.addWidget(table)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return _read_table(table, columns)


def _fill_table(parent: Any, name: str, columns: tuple[str, ...], rows: list[dict[str, str]]) -> Any:
    _dialog, _buttons, QTableWidget, QTableWidgetItem, _layout = _qt_widgets("pop_participantinfo")
    table = QTableWidget(len(rows), len(columns), parent)
    table.setObjectName(name)
    table.setHorizontalHeaderLabels(list(columns))
    for row_index, row in enumerate(rows):
        for column_index, key in enumerate(columns):
            table.setItem(row_index, column_index, QTableWidgetItem(row.get(key, "")))
    return table


def _qt_widgets(function: str) -> tuple[Any, Any, Any, Any, Any]:
    try:
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem, QVBoxLayout
    except ImportError as exc:
        raise ImportError(
            f"{function}: the table dialog needs PySide6. "
            "Install it with `pip install eegprep[gui]`, then open the dialog again."
        ) from exc
    return QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem, QVBoxLayout


def _read_table(table: Any, columns: tuple[str, ...]) -> list[dict[str, str]]:
    collected = []
    for row_index in range(table.rowCount()):
        item: dict[str, str] = {}
        for column_index, key in enumerate(columns):
            cell = table.item(row_index, column_index)
            text = cell.text().strip() if cell is not None else ""
            if text:
                item[key] = text
        collected.append(item)
    return collected


def _store_participant_rows(
    datasets: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    column_info: dict[str, dict[str, str]] | None,
) -> None:
    if len(rows) != len(datasets):
        raise ValueError(
            "pop_participantinfo: the participant table has "
            f"{len(rows)} rows for {len(datasets)} datasets. "
            "Enter one row per dataset, then press Ok."
        )
    for dataset, row in zip(datasets, rows):
        if not str(row.get("participant_id") or "").strip():
            raise ValueError(
                "pop_participantinfo: No subject ID found. Fill in Subject code (participant_id), then press Ok."
            )
        etc = dataset.setdefault("etc", {})
        bids = etc.setdefault("bids", {})
        bids["participant"] = {key: value for key, value in row.items() if str(value).strip()}
        if column_info:
            bids["participant_columns"] = column_info
        dataset["saved"] = "no"


def _event_field_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mapped = {}
    for row in rows:
        name = str(row.get("bids_field") or "").strip()
        if not name:
            continue
        mapped[name] = {
            key: _levels_value(str(row[key])) if key == "Levels" else row[key]
            for key in ("eeglab_field", "Levels", "LongName", "Description", "Units")
            if row.get(key)
        }
    return mapped


def _fan_out(datasets: list[dict[str, Any]] | None, section: str, metadata: dict[str, Any]) -> None:
    for dataset in datasets or []:
        if not isinstance(dataset, dict):
            continue
        etc = dataset.setdefault("etc", {})
        bids = etc.setdefault("bids", {})
        bids.setdefault(section, {}).update(metadata)
        dataset["saved"] = "no"


def _validator_command() -> list[str] | None:
    validator = shutil.which("bids-validator")
    if validator:
        return [validator]
    npx = shutil.which("npx")
    if npx:
        return [npx, "--no-install", "bids-validator"]
    deno = shutil.which("deno")
    if deno:
        return [deno, "run", "-A", "jsr:@bids/validator"]
    return None


def _run_official_validator(path: Path, errors: list[str], warnings: list[str]) -> str | None:
    command = _validator_command()
    if command is None:
        return _VALIDATOR_NOTE
    try:
        completed = subprocess.run(
            [*command, "--json", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _VALIDATOR_NOTE
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if _validator_unavailable(completed.returncode, stdout, stderr):
        return _VALIDATOR_NOTE
    payload = _validator_payload(stdout)
    if payload is None:
        if completed.returncode != 0:
            detail = (stderr or stdout).strip().splitlines()
            message = detail[0] if detail else f"exit {completed.returncode}"
            errors.append(f"validate_bids: bids-validator failed. {message}")
        return None
    issues = payload.get("issues", payload)
    if not isinstance(issues, dict):
        return None
    for issue in issues.get("errors") or []:
        errors.append(f"validate_bids: {_issue_text(issue)}")
    for issue in issues.get("warnings") or []:
        warnings.append(f"validate_bids: {_issue_text(issue)}")
    return None


def _validator_unavailable(code: int, stdout: str, stderr: str) -> bool:
    if stdout.strip().startswith("{"):
        return False
    text = f"{stdout}\n{stderr}".lower()
    missing = (
        "not found",
        "not recognized",
        "npm err",
        "could not determine executable",
        "cannot find module",
        "err_module_not_found",
        "error: could not find",
    )
    return code != 0 and any(marker in text for marker in missing)


def _validator_payload(stdout: str) -> dict[str, Any] | None:
    start = stdout.find("{")
    if start < 0:
        return None
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _issue_text(issue: Any) -> str:
    if isinstance(issue, str):
        return issue
    if isinstance(issue, dict):
        reason = issue.get("reason") or issue.get("key") or issue.get("code") or "bids-validator issue"
        return str(reason)
    return str(issue)


def _dialog_metadata(spec: DialogSpec, renderer: Any, initial: dict[str, Any]) -> dict[str, Any] | None:
    result = inputgui(spec, initial_values=initial or None, renderer=renderer)
    if result is None:
        return None
    metadata: dict[str, Any] = {}
    for key, value in result.items():
        if key == "PowerLineFrequency":
            choice = int(value or 1)
            if choice == 2:
                metadata[key] = "50"
            elif choice == 3:
                metadata[key] = "60"
            continue
        text = str(value or "").strip()
        if text:
            metadata[key] = text
    return metadata


def _task_initial(target: dict[str, Any]) -> dict[str, Any]:
    stored = dict(_section(target, "task"))
    frequency = str(stored.get("PowerLineFrequency") or "")
    if frequency == "50":
        stored["PowerLineFrequency"] = 2
    elif frequency == "60":
        stored["PowerLineFrequency"] = 3
    else:
        stored.pop("PowerLineFrequency", None)
    return stored


def _participant_initial(target: dict[str, Any]) -> dict[str, Any]:
    stored = dict(_section(target, "participant"))
    if not stored.get("participant_id"):
        subject = _subject_text(target.get("subject"))
        if subject:
            stored["participant_id"] = subject
    return stored


def _subject_text(subject: Any) -> str:
    for _ in range(5):
        if isinstance(subject, str) or subject is None:
            break
        shape = getattr(subject, "shape", None)
        if shape == ():
            subject = subject.item()
            break
        if isinstance(subject, (list, tuple)) or shape is not None:
            if len(subject) == 0:
                return ""
            subject = subject[0]
            continue
        break
    text = "" if subject is None else str(subject).strip()
    if not text:
        return ""
    return text if text.startswith("sub-") else f"sub-{text}"


def _section(target: dict[str, Any], name: str) -> dict[str, Any]:
    etc = target.get("etc")
    if not isinstance(etc, dict):
        return {}
    bids = etc.get("bids")
    if not isinstance(bids, dict):
        return {}
    section = bids.get(name)
    return dict(section) if isinstance(section, dict) else {}


def _update_bids_metadata(
    target: dict[str, Any],
    section: str,
    metadata: dict[str, Any],
    command_name: str,
) -> tuple[dict[str, Any], str]:
    etc = target.setdefault("etc", {})
    bids = etc.setdefault("bids", {})
    bids.setdefault(section, {}).update(metadata)
    if command_name == "pop_participantinfo":
        return target, "[EEG, STUDY, command] = pop_participantinfo(EEG, STUDY);"
    target_name = "STUDY" if "datasetinfo" in target else "EEG"
    pieces = [target_name]
    for key, value in metadata.items():
        pieces.extend([format_history_value(key), format_history_value(value)])
    return target, f"LASTCOM = {command_name}({', '.join(pieces)});"
