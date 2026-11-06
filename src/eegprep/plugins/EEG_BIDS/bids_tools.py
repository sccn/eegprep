"""BIDS metadata editors and a file-level dataset check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.plugins.EEG_BIDS.bids_list_eeg_files import bids_list_eeg_files
from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids

_TASK_FIELDS = (
    ("Name", "Dataset name"),
    ("TaskName", "Task name (no space)"),
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


def bids_exporter(EEG: dict[str, Any] | list[dict[str, Any]], output_dir: str | Path) -> tuple[str, str]:
    """Run the EEGPrep BIDS export wizard backend."""
    return pop_exportbids(EEG, output_dir, return_com=True)


def pop_taskinfo(
    target: dict[str, Any],
    *,
    gui: bool = False,
    renderer: Any = None,
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
    return _update_bids_metadata(target, "task", metadata, "pop_taskinfo")


def pop_participantinfo(
    target: dict[str, Any],
    *,
    gui: bool = False,
    renderer: Any = None,
    **metadata: Any,
) -> tuple[dict[str, Any], str]:
    """Attach BIDS participant metadata to an EEG or STUDY dict."""
    if gui or renderer is not None:
        collected = _dialog_metadata(pop_participantinfo_dialog_spec(), renderer, _participant_initial(target))
        if collected is None:
            return target, ""
        if not str(collected.get("participant_id") or "").strip():
            raise ValueError(
                "pop_participantinfo: No subject ID found. Fill in Subject code (participant_id), then press Ok."
            )
        metadata = {**metadata, **collected}
    return _update_bids_metadata(target, "participant", metadata, "pop_participantinfo")


def pop_eventinfo(
    target: dict[str, Any],
    *,
    gui: bool = False,
    renderer: Any = None,
    **metadata: Any,
) -> tuple[dict[str, Any], str]:
    """Attach BIDS event metadata to an EEG or STUDY dict."""
    if gui or renderer is not None:
        collected = _dialog_metadata(pop_eventinfo_dialog_spec(), renderer, _section(target, "event"))
        if collected is None:
            return target, ""
        metadata = {**metadata, **collected}
    return _update_bids_metadata(target, "event", metadata, "pop_eventinfo")


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


def validate_bids(root: str | Path) -> dict[str, list[str]]:
    """Check that a BIDS folder has the sidecars ``pop_exportbids`` writes.

    This does not run the external bids-validator binary used by
    ``pop_validatebids.m``. It does not write a report file into the dataset.
    """
    path = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        errors.append(
            f"validate_bids: BIDS path does not exist: {path}. Pass the folder that contains dataset_description.json."
        )
        return {"errors": errors, "warnings": warnings}
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
    return {"errors": errors, "warnings": warnings}


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
        subject = str(target.get("subject") or "").strip()
        if subject:
            stored["participant_id"] = subject if subject.startswith("sub-") else f"sub-{subject}"
    return stored


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
    target_name = "STUDY" if "datasetinfo" in target else "EEG"
    pieces = [target_name]
    for key, value in metadata.items():
        pieces.extend([format_history_value(key), format_history_value(value)])
    return target, f"LASTCOM = {command_name}({', '.join(pieces)});"
