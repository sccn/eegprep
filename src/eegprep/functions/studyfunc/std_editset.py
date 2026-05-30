"""Edit STUDY set metadata and dataset membership."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._pop_utils import parse_key_value_args, parse_numeric_sequence
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.studyfunc._study_utils import (
    as_alleeg_list,
    build_python_call,
    ensure_study,
    parse_optional_int_text,
    sync_datasetinfo,
)
from eegprep.functions.studyfunc.std_checkset import std_checkset


def std_editset(
    STUDY: dict[str, Any] | None,
    ALLEEG: list[dict[str, Any]] | None,
    *args: Any,
    name: str | None = None,
    task: str | None = None,
    notes: str | None = None,
    filename: str | Path | None = None,
    filepath: str | Path | None = None,
    commands: Any = None,
    updatedat: str = "on",
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Modify STUDY metadata and datasetinfo entries."""
    datasets = [deepcopy(eeg) for eeg in as_alleeg_list(ALLEEG)]
    study = sync_datasetinfo(ensure_study(STUDY), datasets)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    name = options.pop("name", name)
    task = options.pop("task", task)
    notes = options.pop("notes", notes)
    filename = options.pop("filename", filename)
    filepath = options.pop("filepath", filepath)
    commands = options.pop("commands", commands)
    updatedat = str(options.pop("updatedat", updatedat) or "on").lower()
    ignored = {"resave", "savedat", "addchannellabels", "rmclust", "inbrain"}
    unknown = sorted(set(options) - ignored)
    if unknown:
        raise ValueError(f"Unknown std_editset option(s): {', '.join(unknown)}")
    if updatedat not in {"on", "off"}:
        raise ValueError("updatedat must be 'on' or 'off'")

    if name is not None:
        study["name"] = str(name)
    if task is not None:
        study["task"] = str(task)
    if notes is not None:
        study["notes"] = str(notes)
    if filename is not None:
        study["filename"] = Path(filename).name
        if Path(filename).parent != Path("."):
            study["filepath"] = str(Path(filename).parent)
    if filepath is not None:
        study["filepath"] = str(filepath)

    if commands:
        datasets = _apply_commands(study, datasets, commands, update_datasets=updatedat == "on")
    study, datasets = std_checkset(study, datasets)
    command = _history_command(study, commands=commands)
    return (study, datasets, command) if return_com else (study, datasets)


def _apply_commands(
    study: dict[str, Any],
    datasets: list[dict[str, Any]],
    commands: Any,
    *,
    update_datasets: bool,
) -> list[dict[str, Any]]:
    flat = _flatten_commands(commands)
    current_index = 1
    info = list(study.get("datasetinfo") or [])
    index = 0
    while index < len(flat):
        key = str(flat[index]).lower()
        value = flat[index + 1] if index + 1 < len(flat) else None
        if key == "index":
            current_index = int(value)
        elif key == "load":
            eeg = pop_loadset(str(value))
            current_index = _bounded_insert_index(current_index, len(datasets))
            if current_index <= len(datasets):
                datasets[current_index - 1] = eeg
            else:
                datasets.append(eeg)
            study["datasetinfo"] = info = sync_datasetinfo(study, datasets)["datasetinfo"]
        elif key == "remove":
            remove_index = int(value)
            if remove_index < 1 or remove_index > len(datasets):
                raise ValueError("remove index is outside ALLEEG")
            del datasets[remove_index - 1]
            del info[remove_index - 1]
            study["datasetinfo"] = info
        else:
            if current_index < 1 or current_index > len(info):
                raise ValueError("command index is outside STUDY.datasetinfo")
            normalized = _metadata_value(key, value)
            info[current_index - 1][key] = normalized
            if update_datasets and current_index <= len(datasets):
                datasets[current_index - 1][key] = normalized
        index += 2
    study["datasetinfo"] = info
    return datasets


def _flatten_commands(commands: Any) -> list[Any]:
    if isinstance(commands, dict):
        flat: list[Any] = []
        for key, value in commands.items():
            flat.extend([key, value])
        return flat
    if isinstance(commands, tuple):
        commands = list(commands)
    if not isinstance(commands, list):
        raise ValueError("commands must be a list, tuple, or dict")
    flat = []
    for item in commands:
        if isinstance(item, dict):
            flat.extend(_flatten_commands(item))
        elif isinstance(item, (list, tuple)):
            flat.extend(_flatten_commands(list(item)))
        else:
            flat.append(item)
    if len(flat) % 2:
        raise ValueError("commands must contain key/value pairs")
    return flat


def _metadata_value(key: str, value: Any) -> Any:
    if key in {"session", "run"}:
        return parse_optional_int_text(value)
    if key == "comps":
        if value in (None, ""):
            return []
        return parse_numeric_sequence(value, dtype=int)
    return "" if value is None else value


def _bounded_insert_index(index: int, length: int) -> int:
    if index < 1:
        raise ValueError("dataset index must be 1-based")
    return min(index, length + 1)


def _history_command(study: dict[str, Any], *, commands: Any) -> str:
    kwargs: dict[str, Any] = {
        "name": study.get("name", ""),
        "task": study.get("task", ""),
        "notes": study.get("notes", ""),
        "filename": study.get("filename", ""),
        "filepath": study.get("filepath", ""),
    }
    if commands:
        kwargs["commands"] = commands
    return build_python_call(("STUDY", "ALLEEG"), "std_editset", "STUDY", "ALLEEG", **kwargs)


__all__ = ["std_editset"]
