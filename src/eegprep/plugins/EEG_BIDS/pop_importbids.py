"""Import EEG data from a BIDS folder or BIDS EEG file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.extras import import_optional_dependency
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.studyfunc._study_utils import ensure_study
from eegprep.plugins.EEG_BIDS.bids_list_eeg_files import bids_list_eeg_files
from eegprep.plugins.EEG_BIDS.sidecar import read_bids_table

_TYPE_FIELDS = ("value", "trial_type", "type", "event_type", "HED", "code")
_DIALOG_KEYS = {
    "gui",
    "renderer",
    "return_study",
    "studyname",
    "outputdir",
    "subjects",
    "sessions",
    "runs",
    "tasks",
    "bidstask",
    "recordings",
}


def pop_importbids(
    path: str | Path,
    *,
    return_com: bool = False,
    gui: bool = False,
    renderer: Any = None,
    return_study: bool = False,
    **kwargs: Any,
) -> Any:
    """Load one or more EEG files from a BIDS dataset.

    EDF and BDF files are read with ``pop_biosig``. TSV sidecars are decoded
    as UTF-8, then as Latin-1 when the bytes are not UTF-8. JSON sidecars
    stay UTF-8, and a leading UTF-8 BOM is removed.

    ``gui=True`` shows the options dialog from ``pop_importbids.m`` lines
    104-164. ``return_study=True`` returns an in-memory STUDY. EEGLAB also
    copies every dataset into ``derivatives/eeglab`` (``pop_importbids.m``
    lines 839 and 892). This function does not write that copy.
    """
    source = Path(path)
    dialog_options: dict[str, Any] = {}
    if gui and source.is_dir():
        selected = _import_dialog(source, renderer)
        if selected is None:
            if return_study and return_com:
                return None, None, ""
            if return_com:
                return None, ""
            return None
        dialog_options = selected
    loader_options = _loader_options(kwargs, dialog_options)
    list_filters = _list_filters(kwargs, dialog_options)
    if source.is_dir():
        import_optional_dependency("bids", feature="pop_importbids")
        files = bids_list_eeg_files(str(source), **list_filters)
    else:
        files = [str(source)]
    if not files:
        raise ValueError(
            f"pop_importbids: no supported EEG files found in {source}. "
            "Point it at a BIDS folder that contains .set, .edf, .bdf, or .vhdr files."
        )
    eegs = [pop_load_frombids(filename, **loader_options) for filename in files]
    study_name = str(dialog_options.get("studyName") or kwargs.get("studyName") or source.name)
    command = _history_command(source, dialog_options, return_study=return_study)
    if return_study:
        study = ensure_study({"name": study_name, "task": str(dialog_options.get("bidstask") or "")})
        study, datasets = std_checkset(study, eegs)
        for dataset in datasets:
            dataset["history"] = command
        if return_com:
            return study, datasets, command
        return study, datasets
    result: dict[str, Any] | list[dict[str, Any]] = eegs[0] if len(eegs) == 1 else eegs
    if isinstance(result, list):
        for eeg in result:
            eeg["history"] = command
    else:
        result["history"] = command
    return (result, command) if return_com else result


def pop_importbids_dialog_spec(path: str | Path, type_fields: tuple[str, ...] = _TYPE_FIELDS) -> DialogSpec:
    """Return the import options dialog from ``pop_importbids.m`` lines 104-125."""
    folder = Path(path)
    fields = type_fields or ("value",)
    browse = CallbackSpec(
        "select_file",
        {
            "button": "browse",
            "target": "folder",
            "mode": "directory",
            "caption": "Study output folder",
        },
    )
    return DialogSpec(
        title="Import BIDS data -- pop_importbids()",
        function_name="pop_importbids",
        eeglab_source="plugins/EEG-BIDS/pop_importbids.m",
        geometry=(
            [2, 1.5],
            [1],
            [1],
            [1, 0.35],
            [0.6, 0.35, 0.5],
            [0.6, 0.35, 0.5],
            [0.6, 0.35, 0.5],
            [0.6, 0.35, 0.5],
            [1],
            [1, 2, 0.5],
        ),
        controls=(
            ControlSpec("text", "Enter study name (default is BIDS folder name)"),
            ControlSpec("edit", tag="studyName"),
            ControlSpec("text", ""),
            ControlSpec(
                "checkbox",
                "Use BIDS electrode.tsv files (when present) for channel locations; off: look up locations using channel labels",
                tag="chanlocs",
                value=True,
            ),
            ControlSpec(
                "checkbox",
                "Use BIDS event.tsv files for events and use the following BIDS field for event type",
                tag="events",
                value=True,
            ),
            ControlSpec("popupmenu", "|".join(fields), tag="typefield", value=1),
            ControlSpec("checkbox", "Import only the following BIDS tasks", tag="bidstask", value=False),
            ControlSpec("edit", tag="bidstaskstr"),
            ControlSpec("text", ""),
            ControlSpec("checkbox", "Import only the following sessions", tag="bidssessions", value=False),
            ControlSpec("edit", tag="bidsessionstr"),
            ControlSpec("text", ""),
            ControlSpec("checkbox", "Import only the following runs", tag="bidsruns", value=False),
            ControlSpec("edit", tag="bidsrunsstr"),
            ControlSpec("text", ""),
            ControlSpec("checkbox", "Import only the following participant indices", tag="bidssubjects", value=False),
            ControlSpec("edit", tag="bidssubjectsstr"),
            ControlSpec("text", ""),
            ControlSpec("text", ""),
            ControlSpec("text", "Study output folder"),
            ControlSpec("edit", str(folder / "derivatives" / "eeglab"), tag="folder"),
            ControlSpec("pushbutton", "...", tag="browse", callback=browse),
        ),
        known_differences=(
            "The output folder is recorded on the command but datasets stay in memory. "
            "pop_importbids.m lines 839 and 892 also save a derivatives/eeglab copy.",
            "Task, session, and run filters are text fields. pop_importbids.m lines 112-118 "
            "fills a task popup and session/run listboxes by scanning the folder. The typed "
            "identifiers are the same values the import filter accepts. The recordings row "
            "(lines 117-118) is shown only when a recording entity exists, so it is omitted here.",
        ),
    )


def _import_dialog(source: Path, renderer: Any) -> dict[str, Any] | None:
    fields = _event_columns(source)
    result = inputgui(pop_importbids_dialog_spec(source, fields), renderer=renderer)
    if not result:
        return None
    options: dict[str, Any] = {}
    study_name = str(result.get("studyName") or "").strip()
    if study_name:
        options["studyName"] = study_name
    options["bidschanloc"] = bool(result.get("chanlocs", True))
    options["bidsevent"] = bool(result.get("events", True))
    choice = int(result.get("typefield") or 1)
    if 1 <= choice <= len(fields):
        options["eventtype"] = fields[choice - 1]
    if result.get("bidstask"):
        task = str(result.get("bidstaskstr") or "").strip()
        if task:
            options["bidstask"] = task
    if result.get("bidssessions"):
        sessions = _split_list(result.get("bidsessionstr"))
        if sessions:
            options["sessions"] = sessions
    if result.get("bidsruns"):
        runs = _split_list(result.get("bidsrunsstr"))
        if runs:
            options["runs"] = runs
    if result.get("bidssubjects"):
        subjects = _split_list(result.get("bidssubjectsstr"))
        if subjects:
            options["subjects"] = subjects
    folder = str(result.get("folder") or "").strip()
    if folder:
        options["outputdir"] = folder
    return options


def _event_columns(source: Path) -> tuple[str, ...]:
    found: list[str] = []
    for events in source.rglob("*_events.tsv"):
        try:
            table = read_bids_table(events)
        except (OSError, UnicodeError, ValueError):
            continue
        for column in table.columns:
            if column not in found:
                found.append(str(column))
        break
    ordered = [name for name in _TYPE_FIELDS if name in found]
    ordered.extend(name for name in found if name not in ordered and name not in {"onset", "duration", "sample"})
    if "value" in ordered:
        ordered.insert(0, ordered.pop(ordered.index("value")))
    return tuple(ordered or _TYPE_FIELDS)


def _loader_options(kwargs: dict[str, Any], dialog: dict[str, Any]) -> dict[str, Any]:
    options = {key: value for key, value in kwargs.items() if key.lower() not in _DIALOG_KEYS}
    if "bidschanloc" in dialog:
        options["bidschanloc"] = dialog["bidschanloc"]
    if "bidsevent" in dialog:
        options["bidsevent"] = "replace" if dialog["bidsevent"] else False
    if dialog.get("eventtype"):
        options["eventtype"] = dialog["eventtype"]
    return options


def _list_filters(kwargs: dict[str, Any], dialog: dict[str, Any]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    subjects = dialog.get("subjects", kwargs.get("subjects"))
    sessions = dialog.get("sessions", kwargs.get("sessions"))
    runs = dialog.get("runs", kwargs.get("runs"))
    tasks = dialog.get("bidstask", kwargs.get("bidstask", kwargs.get("tasks")))
    if subjects:
        filters["subjects"] = subjects
    if sessions:
        filters["sessions"] = sessions
    if runs:
        filters["runs"] = runs
    if tasks:
        filters["tasks"] = tasks
    return filters


def _split_list(value: Any) -> list[str]:
    text = str(value or "").replace(",", " ")
    return [item for item in text.split() if item]


def _history_command(source: Path, dialog: dict[str, Any], *, return_study: bool) -> str:
    # pop_importbids.m:895 records the dialog options with vararg2str.
    parts = [format_history_value(source)]
    if dialog.get("eventtype"):
        parts.extend([format_history_value("eventtype"), format_history_value(dialog["eventtype"])])
    if "bidsevent" in dialog:
        parts.extend([format_history_value("bidsevent"), "'on'" if dialog["bidsevent"] else "'off'"])
    if "bidschanloc" in dialog:
        parts.extend([format_history_value("bidschanloc"), "'on'" if dialog["bidschanloc"] else "'off'"])
    if dialog.get("studyName"):
        parts.extend([format_history_value("studyName"), format_history_value(dialog["studyName"])])
    if dialog.get("outputdir"):
        parts.extend([format_history_value("outputdir"), format_history_value(Path(dialog["outputdir"]))])
    for key in ("bidstask", "sessions", "runs", "subjects"):
        if dialog.get(key):
            parts.extend([format_history_value(key), format_history_value(dialog[key])])
    call = f"pop_importbids({', '.join(parts)})"
    if return_study:
        return f"[STUDY, ALLEEG] = {call};"
    return f"EEG = {call};"
