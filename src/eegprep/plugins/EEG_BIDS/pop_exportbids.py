"""Export EEGPrep datasets to a BIDS EEG folder."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._file_io import channel_labels, events_to_records, write_json
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_saveset import pop_saveset


def pop_exportbids(
    EEG: dict[str, Any] | list[dict[str, Any]],
    output_dir: str | Path | None = None,
    *,
    subject: str = "01",
    task: str = "eeg",
    license: str = "CC0",  # noqa: A002
    changes: str = "",
    createids: bool | str = False,
    return_com: bool = False,
    gui: bool = False,
    renderer: Any = None,
    require_study: bool = False,
    study: dict[str, Any] | None = None,
) -> str | tuple[str, str]:
    """Export EEG dataset(s) as BIDS EEGLAB files and sidecars.

    The File menu passes ``require_study=True``. EEGLAB rejects that call when
    no study is loaded (``pop_exportbids.m`` line 33) because it copies files
    that were already saved (line 204). A direct ``pop_exportbids(EEG, folder)``
    call still writes the dataset from memory so the command-line export keeps
    working.

    Each dataset is filed under ``EEG['subject']``, or the matching
    ``STUDY.datasetinfo`` subject when that field is empty. A single dataset
    with neither set uses ``subject``. ``createids`` replaces the unique
    subjects with ``sub-001``, ``sub-002``, and so on. Session and run come
    from the dataset when they are set.
    """
    datasets = [item for item in (EEG if isinstance(EEG, list) else [EEG]) if isinstance(item, dict)]
    if require_study and not study:
        raise ValueError(
            "pop_exportbids: BIDS export can only export EEGLAB studies. "
            "Create a study with File > Create study, then export again."
        )
    license_name = str(license)
    changes_text = str(changes)
    createids_flag = _createids_flag(createids)
    if gui or (output_dir is None and renderer is not None):
        selected = _export_dialog(datasets, renderer, task=str((study or {}).get("task") or task))
        if selected is None:
            return ("", "") if return_com else ""
        output_dir, task, license_name, changes_text, createids_flag = selected
    if output_dir is None:
        raise ValueError(
            "pop_exportbids: an output folder is required. "
            "Call pop_exportbids(EEG, output_dir) or open the export dialog with gui=True."
        )
    _check_task_name(task)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    task_info = _task_info(datasets, task)
    _write_description(root / "dataset_description.json", task_info, license_name)
    _write_readme(root / "README", task_info)
    if changes_text:
        (root / "CHANGES").write_text(changes_text.rstrip() + "\n", encoding="utf-8")
    labels, used_subject = _subject_labels(datasets, study, subject=subject)
    exported_labels = _write_recordings(
        root,
        datasets,
        labels,
        study=study,
        task=task,
        createids=createids_flag,
        task_info=task_info,
    )
    _write_participants(root / "participants.tsv", datasets, exported_labels)
    command = _history_command(
        root,
        datasets,
        task=task,
        license_name=license_name,
        changes=changes_text,
        createids=createids_flag,
        subject=used_subject,
    )
    return (str(root), command) if return_com else str(root)


def pop_exportbids_dialog_spec(*, task: str = "eeg") -> DialogSpec:
    """Return the study export wizard from ``pop_exportbids.m`` lines 48-83."""
    browse = CallbackSpec(
        "select_file",
        {
            "button": "browse",
            "target": "outputfolder",
            "mode": "directory",
            "caption": "Pick a BIDS output folder",
        },
    )
    return DialogSpec(
        title="Export EEGLAB STUDY to BIDS -- pop_exportbids()",
        function_name="pop_exportbids",
        eeglab_source="plugins/EEG-BIDS/pop_exportbids.m",
        geometry=([1], [1], [0.45, 0.45, 0.2], [1], [1], [1], [1], [1, 1, 1], [1], [0.6, 0.4]),
        controls=(
            ControlSpec("text", "Export EEGLAB study to BIDS", font_weight="bold"),
            ControlSpec("text", ""),
            ControlSpec("text", "Output folder:"),
            ControlSpec("edit", tag="outputfolder", value="BIDS_EXPORT"),
            ControlSpec("pushbutton", "...", tag="browse", callback=browse),
            ControlSpec("text", "Licence for distributing:"),
            ControlSpec("edit", tag="license", value="Creative Common 0 (CC0)"),
            ControlSpec("text", "CHANGES compared to previous releases:"),
            ControlSpec("edit", tag="changes"),
            ControlSpec("pushbutton", "Edit task & EEG info", tag="task"),
            ControlSpec("pushbutton", "Edit participants", tag="participants"),
            ControlSpec("pushbutton", "Edit event info", tag="events"),
            ControlSpec(
                "checkbox",
                "Do not use participants IDs and create anonymized participants IDs instead",
                tag="newids",
                value=False,
            ),
            ControlSpec("text", "Task name (no space or underscore)"),
            ControlSpec("edit", tag="taskname", value=task),
        ),
        known_differences=(
            "Task name is on this dialog. EEGLAB asks in a second window when STUDY.task is empty "
            "(pop_exportbids.m lines 90-98).",
            "PowerLineFrequency defaults to n/a, the value bids_writetinfofile.m line 70 writes. "
            "EEGLAB prompts when the field is missing (pop_exportbids.m lines 100-113).",
            "The dialog records a replayable pop_exportbids command with the dataset list, task, "
            "license, changes, and createids. pop_exportbids.m lines 247-252 leave that command empty.",
        ),
    )


def _export_dialog(
    datasets: list[dict[str, Any]],
    renderer: Any,
    *,
    task: str,
) -> tuple[Path, str, str, str, bool] | None:
    spec = pop_exportbids_dialog_spec(task=task)
    if renderer is None:
        result = _run_qt_export_dialog(spec, datasets)
    else:
        result = inputgui(spec, renderer=renderer)
    if not result:
        return None
    folder = str(result.get("outputfolder") or "").strip()
    if not folder:
        raise ValueError(
            "pop_exportbids: an output folder is required. Choose a folder in the export dialog, then press Ok."
        )
    task_name = str(result.get("taskname") or task).strip()
    if not task_name:
        raise ValueError("pop_exportbids: a task name is required. Enter a task name with no spaces or underscores.")
    _check_task_name(task_name)
    license_name = str(result.get("license") or "CC0")
    if "CC0" in license_name:
        license_name = "CC0"
    return Path(folder), task_name, license_name, str(result.get("changes") or ""), bool(result.get("newids"))


def _run_qt_export_dialog(spec: DialogSpec, datasets: list[dict[str, Any]]) -> dict[str, Any] | None:
    from eegprep.functions.guifunc.qt import QtDialogRenderer
    from eegprep.plugins.EEG_BIDS.bids_tools import pop_eventinfo, pop_participantinfo, pop_taskinfo

    _app, dialog, widgets = inputgui(spec, mode="plot")
    holder = {"datasets": datasets}

    def _edit(editor) -> None:
        current = holder["datasets"]
        updated, command = editor(current[0], gui=True)
        if command:
            current[0].clear()
            current[0].update(updated)

    for tag, editor in (
        ("task", pop_taskinfo),
        ("participants", pop_participantinfo),
        ("events", pop_eventinfo),
    ):
        button = widgets.get(tag)
        if button is not None and hasattr(button, "clicked"):
            button.clicked.connect(lambda _checked=False, editor=editor: _edit(editor))
    if dialog.exec() != QtDialogRenderer._QDialog().Accepted:
        return None
    return {tag: QtDialogRenderer._read_widget(widget) for tag, widget in widgets.items()}


def _check_task_name(task: str) -> None:
    if any(character in task for character in (" ", "_")):
        raise ValueError(
            f"pop_exportbids: task name {task!r} contains a space or underscore. "
            "Rename the task so it can be a BIDS task- entity, then export again."
        )


def _task_info(datasets: list[dict[str, Any]], task: str) -> dict[str, Any]:
    stored: dict[str, Any] = {}
    for dataset in datasets:
        etc = dataset.get("etc")
        if isinstance(etc, dict):
            bids = etc.get("bids")
            if isinstance(bids, dict) and isinstance(bids.get("task"), dict):
                stored.update(bids["task"])
    info = dict(stored)
    info["TaskName"] = task
    info.setdefault("EEGReference", _reference(datasets[0]) if datasets else "Unknown")
    info.setdefault("PowerLineFrequency", "n/a")
    info.setdefault("SoftwareFilters", "n/a")
    return info


def _reference(eeg: dict[str, Any]) -> str:
    ref = eeg.get("ref")
    if isinstance(ref, str) and ref.strip():
        return ref
    return "Unknown"


def _write_description(path: Path, task_info: dict[str, Any], license_name: str) -> None:
    payload: dict[str, Any] = {
        "Name": str(task_info.get("Name") or "EEGPrep export"),
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "License": license_name,
    }
    if task_info.get("Authors"):
        payload["Authors"] = [item.strip() for item in str(task_info["Authors"]).split(";") if item.strip()]
    write_json(path, payload)


def _write_readme(path: Path, task_info: dict[str, Any]) -> None:
    name = str(task_info.get("Name") or "EEGPrep export")
    intro = str(task_info.get("README") or "EEG dataset exported by EEGPrep.").strip()
    path.write_text(f"# {name}\n\n{intro}\n", encoding="utf-8")


def _createids_flag(value: bool | str) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "on":
            return True
        if text == "off":
            return False
        raise ValueError(
            f"pop_exportbids: createids {value!r} is not 'on' or 'off'. "
            "Pass createids='on' to anonymize subject IDs, or createids='off' to keep them."
        )
    return bool(value)


def _bids_label(value: Any) -> str:
    from eegprep.plugins.EEG_BIDS.bids_tools import _subject_text

    text = _subject_text(value)
    if text.lower().startswith("sub-"):
        text = text[4:]
    return text.strip()


def _entity_label(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return ""
        if float(value).is_integer():
            value = int(value)
    text = _bids_label(value)
    lowered = text.lower()
    for prefix in ("ses-", "run-"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            break
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def _dataset_info(study: dict[str, Any] | None, count: int) -> list[dict[str, Any]]:
    raw = (study or {}).get("datasetinfo") or []
    if isinstance(raw, dict):
        raw = [raw]
    infos: list[dict[str, Any]] = []
    for index in range(count):
        item = raw[index] if index < len(raw) and isinstance(raw[index], dict) else {}
        infos.append(item)
    return infos


def _subject_labels(
    datasets: list[dict[str, Any]],
    study: dict[str, Any] | None,
    *,
    subject: str,
) -> tuple[list[str], str | None]:
    # pop_exportbids.m:167-172 reads EEG.subject, then STUDY.datasetinfo.subject.
    allow_argument = study is None and len(datasets) == 1
    infos = _dataset_info(study, len(datasets))
    labels: list[str] = []
    used_argument = False
    for index, (eeg, info) in enumerate(zip(datasets, infos), start=1):
        label = _bids_label(eeg.get("subject")) or _bids_label(info.get("subject"))
        if not label and allow_argument:
            label = _bids_label(subject)
            used_argument = bool(label)
        if not label:
            raise ValueError(
                f"pop_exportbids: no subject info found for dataset {index} in either EEG or "
                "STUDY.datasetinfo. Add a subject with Study > Edit STUDY info, or set "
                "EEG['subject'], then export again."
            )
        labels.append(label)
    return labels, subject if used_argument else None


def _history_command(
    root: Path,
    datasets: list[dict[str, Any]],
    *,
    task: str,
    license_name: str,
    changes: str,
    createids: bool,
    subject: str | None,
) -> str:
    parts = ["EEG" if len(datasets) == 1 else "ALLEEG", format_history_value(root)]
    if subject is not None:
        parts.extend([format_history_value("subject"), format_history_value(subject)])
    parts.extend(
        [
            format_history_value("task"),
            format_history_value(task),
            format_history_value("license"),
            format_history_value(license_name),
            format_history_value("changes"),
            format_history_value(changes),
            format_history_value("createids"),
            "'on'" if createids else "'off'",
        ]
    )
    return f"LASTCOM = pop_exportbids({', '.join(parts)});"


def _exported_subject_ids(labels: list[str], *, createids: bool) -> list[str]:
    # bids_export.m:605-608 numbers unique subjects, so one subject's sessions share an id.
    unique: list[str] = []
    for label in labels:
        if label not in unique:
            unique.append(label)
    if not createids:
        return list(labels)
    aliases = {label: f"{index:03d}" for index, label in enumerate(unique, start=1)}
    return [aliases[label] for label in labels]


def _write_recordings(
    root: Path,
    datasets: list[dict[str, Any]],
    labels: list[str],
    *,
    study: dict[str, Any] | None,
    task: str,
    createids: bool,
    task_info: dict[str, Any],
) -> list[str]:
    exported = _exported_subject_ids(labels, createids=createids)
    infos = _dataset_info(study, len(datasets))
    run_counts: dict[str, int] = {}
    for eeg, info, subject_label in zip(datasets, infos, exported):
        session = _entity_label(eeg.get("session")) or _entity_label(info.get("session"))
        run = _entity_label(eeg.get("run")) or _entity_label(info.get("run"))
        if not run:
            run_counts[subject_label] = run_counts.get(subject_label, 0) + 1
            run = f"{run_counts[subject_label]:02d}"
        eeg_dir = root / f"sub-{subject_label}"
        if session:
            eeg_dir = eeg_dir / f"ses-{session}"
        eeg_dir = eeg_dir / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"sub-{subject_label}"
        if session:
            prefix += f"_ses-{session}"
        prefix += f"_task-{task}_run-{run}"
        target = eeg_dir / f"{prefix}_eeg.set"
        if target.exists():
            raise ValueError(
                f"pop_exportbids: {prefix}_eeg.set already exists. Set a distinct EEG['session'] "
                "or EEG['run'] on each dataset, then export again."
            )
        pop_saveset(deepcopy(eeg), str(target))
        _write_channels(eeg_dir / f"{prefix}_channels.tsv", eeg)
        _write_events(eeg_dir / f"{prefix}_events.tsv", eeg)
        _write_eeg_json(eeg_dir / f"{prefix}_eeg.json", eeg, task_info)
        electrode_stem = _electrode_basename(prefix)
        _write_electrodes(eeg_dir / f"{electrode_stem}_electrodes.tsv", eeg)
        _write_coordsystem(eeg_dir / f"{electrode_stem}_coordsystem.json", eeg)
    return exported


def _write_participants(path: Path, datasets: list[dict[str, Any]], subjects: list[str]) -> None:
    extra_fields = ("Gender", "Age", "Group", "HeadCircumference", "SubjectArtefactDescription")
    rows = []
    seen: set[str] = set()
    for dataset, subject_label in zip(datasets, subjects):
        if subject_label in seen:
            continue
        seen.add(subject_label)
        participant = _section(dataset, "participant")
        row = {"participant_id": f"sub-{subject_label}"}
        for field in extra_fields:
            if field in participant and participant[field] not in ("", None):
                row[field] = participant[field]
        rows.append(row)
    fields = ["participant_id"]
    for field in extra_fields:
        if any(field in row for row in rows):
            fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "n/a") for field in fields})


def _write_channels(path: Path, eeg: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["name", "type", "units"])
        for label in channel_labels(eeg):
            writer.writerow([label, "EEG", "uV"])


def _write_events(path: Path, eeg: dict[str, Any]) -> None:
    events = events_to_records(eeg.get("event"))
    include_trial_type = any("trial_type" in event for event in events)
    fields = ["onset", "duration", "sample", "value"]
    if include_trial_type:
        fields.append("trial_type")
    srate = float(eeg.get("srate") or 1.0)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(fields)
        for event in events:
            latency = float(event.get("latency", 1) or 1)
            onset = (latency - 1) / srate
            row = [
                _bids_number(onset),
                _duration_cell(event, srate),
                _sample_cell(event),
                _value_cell(event),
            ]
            if include_trial_type:
                row.append(_text_cell(event.get("trial_type")))
            writer.writerow(row)


def _write_eeg_json(path: Path, eeg: dict[str, Any], task_info: dict[str, Any]) -> None:
    trials = int(eeg.get("trials") or 1)
    pnts = int(eeg.get("pnts") or np.asarray(eeg.get("data")).shape[-1])
    srate = float(eeg.get("srate") or 1.0)
    duration = pnts / srate
    payload: dict[str, Any] = {
        "TaskName": task_info["TaskName"],
        "SamplingFrequency": srate,
        "EEGReference": task_info.get("EEGReference") or "Unknown",
        "PowerLineFrequency": task_info.get("PowerLineFrequency", "n/a"),
        "SoftwareFilters": task_info.get("SoftwareFilters", "n/a"),
        "EEGChannelCount": int(eeg.get("nbchan") or np.asarray(eeg.get("data")).shape[0]),
        "RecordingType": "continuous" if trials == 1 else "epoched",
        "RecordingDuration": duration if trials == 1 else duration * trials,
    }
    if trials != 1:
        payload["EpochLength"] = duration
    for key in (
        "TaskDescription",
        "Instructions",
        "CogAtlasID",
        "CogPOID",
        "InstitutionName",
        "InstitutionAddress",
        "InstitutionalDepartmentName",
        "DeviceSerialNumber",
        "EEGGround",
        "EEGPlacementScheme",
        "Manufacturer",
        "ManufacturersModelName",
        "CapManufacturer",
        "CapManufacturersModelName",
        "HardwareFilters",
        "SoftwareVersions",
    ):
        if task_info.get(key) not in (None, ""):
            payload[key] = task_info[key]
    write_json(path, payload)


def _electrode_basename(stem: str) -> str:
    # bids_writeelectrodefile.m:32-49 removes task- and desc- from the electrodes name.
    return _drop_filename_entity(_drop_filename_entity(stem, "task-"), "desc-")


def _drop_filename_entity(name: str, token: str) -> str:
    start = name.find(token)
    if start < 0:
        return name
    end = name.find("_", start)
    if end < 0:
        end = len(name)
    cut = start - 1 if start > 0 and name[start - 1] == "_" else start
    return name[:cut] + name[end:]


def _write_coordsystem(path: Path, eeg: dict[str, Any]) -> None:
    # bids_writeelectrodefile.m:272-316 writes one coordsystem.json beside electrodes.tsv.
    chaninfo = eeg.get("chaninfo")
    bids = chaninfo.get("BIDS") if isinstance(chaninfo, dict) else None
    if not isinstance(bids, dict):
        bids = {}
    systems = bids.get("coordsystems")
    if isinstance(systems, list) and systems:
        base = path.name.removesuffix("_coordsystem.json")
        for system in systems:
            if not isinstance(system, dict):
                continue
            payload = {key: value for key, value in system.items() if str(key).lower() != "space"}
            space = str(system.get("space") or "")
            target = path.with_name(f"{base}_space-{space}_coordsystem.json") if space else path
            write_json(target, payload)
        return
    etc = eeg.get("etc")
    datatype = str(etc.get("datatype") or "") if isinstance(etc, dict) else ""
    if datatype.lower() == "emg":
        payload = {
            "EMGCoordinateUnits": bids.get("EMGCoordinateUnits", "mm"),
            "EMGCoordinateSystem": bids.get("EMGCoordinateSystem", "Other"),
            "EMGCoordinateSystemDescription": bids.get("EMGCoordinateSystemDescription", "Electrode locations in mm"),
        }
    else:
        payload = {
            "EEGCoordinateUnits": bids.get("EEGCoordinateUnits", "mm"),
            "EEGCoordinateSystem": bids.get("EEGCoordinateSystem", "CTF"),
            "EEGCoordinateSystemDescription": bids.get("EEGCoordinateSystemDescription", "EEGLAB"),
        }
    write_json(path, payload)


def _write_electrodes(path: Path, eeg: dict[str, Any]) -> None:
    chanlocs = eeg.get("chanlocs")
    if isinstance(chanlocs, np.ndarray):
        chanlocs = [] if chanlocs.size == 0 else chanlocs.tolist()
    elif not chanlocs:
        chanlocs = []
    labels = channel_labels(eeg)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["name", "x", "y", "z"])
        for index, label in enumerate(labels):
            loc = chanlocs[index] if index < len(chanlocs) and isinstance(chanlocs[index], dict) else {}
            writer.writerow([label, _coord(loc, "X", "x"), _coord(loc, "Y", "y"), _coord(loc, "Z", "z")])


def _section(eeg: dict[str, Any], name: str) -> dict[str, Any]:
    etc = eeg.get("etc")
    if not isinstance(etc, dict):
        return {}
    bids = etc.get("bids")
    if not isinstance(bids, dict):
        return {}
    section = bids.get(name)
    return section if isinstance(section, dict) else {}


def _duration_cell(event: dict[str, Any], srate: float) -> str:
    if "duration" not in event or event["duration"] in ("", None):
        return "n/a"
    value = event["duration"]
    if isinstance(value, str):
        return "n/a" if value.strip().lower() in {"", "nan", "n/a"} else value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return _bids_number(number / srate)


def _sample_cell(event: dict[str, Any]) -> str:
    # bids_writeeventfile.m:197 stores latency-1.
    latency = event.get("latency")
    try:
        number = float(latency)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return str(int(round(number)) - 1)


def _value_cell(event: dict[str, Any]) -> str:
    source = event["value"] if "value" in event and event["value"] not in ("", None) else event.get("type", "")
    return _text_cell(source)


def _text_cell(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "n/a"
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.10g}"
    text = str(value).strip()
    if text.lower() == "nan":
        return "n/a"
    return text


def _bids_number(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{float(value):.10g}"


def _coord(loc: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key not in loc or loc[key] in ("", None):
            continue
        try:
            number = float(loc[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return f"{number:.10g}"
    return "n/a"
