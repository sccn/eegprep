"""EEGLAB-style dataset switching/storing helper."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
from eegprep.functions.adminfunc.eeg_store import eeg_store
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.popfunc.pop_saveset import pop_saveset


def pop_newset(
    ALLEEG: list[dict[str, Any]] | None,
    EEG: dict[str, Any] | list[dict[str, Any]],
    CURRENTSET: int | list[int] | tuple[int, ...] | None,
    *args: Any,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | list[dict[str, Any]], int | list[int], str]:
    """Store or retrieve datasets following EEGLAB's ``pop_newset`` contract."""
    renderer = kwargs.pop("renderer", None)
    options = parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True)
    allowed = {
        "retrieve",
        "setname",
        "comments",
        "overwrite",
        "saveold",
        "savenew",
        "gui",
        "guistring",
        "study",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"Unsupported pop_newset option(s): {', '.join(unknown)}")

    alleeg = [] if ALLEEG is None else list(ALLEEG)
    retrieve = options.get("retrieve")
    if retrieve is not None and retrieve != []:
        current, alleeg, current_set = eeg_retrieve(alleeg, retrieve)
        command = _history_command({"retrieve": retrieve})
        return alleeg, current, current_set, command

    if _is_on(options.get("gui", False)) and isinstance(EEG, dict):
        gui_result = _run_gui(EEG, CURRENTSET, options, renderer=renderer)
        if gui_result is None:
            current, alleeg, current_set = eeg_retrieve(alleeg, CURRENTSET or 1)
            return alleeg, current, current_set, ""
        options.update(gui_result)

    eeg_to_store = _apply_dataset_metadata(EEG, options)
    if _should_save(options.get("saveold", False)) and CURRENTSET:
        _save_existing_datasets(alleeg, CURRENTSET)
    saved_new = _should_save(options.get("savenew", False))
    if saved_new:
        _save_new_dataset(eeg_to_store, options.get("savenew"))

    store_index = _store_index(CURRENTSET, eeg_to_store, options)
    alleeg, current, current_set = eeg_store(alleeg, eeg_to_store, store_index)
    if saved_new and isinstance(current, dict) and isinstance(current_set, int):
        current["saved"] = "yes"
        alleeg[current_set - 1]["saved"] = "yes"
    command = _history_command(_command_options(options, store_index))
    return alleeg, current, current_set, command


def pop_newset_dialog_spec(EEG: dict[str, Any], CURRENTSET: Any = None, *, guistring: str = "") -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_newset``."""
    dataset_name = str(EEG.get("setname") or "")
    prompt = guistring or "What do you want to do with the new dataset?"
    old_prompt = "What do you want to do with the old dataset (not modified since last saved)?"
    return DialogSpec(
        title="Dataset info -- pop_newset()",
        function_name="pop_newset",
        eeglab_source="functions/popfunc/pop_newset.m",
        geometry=((1,), (0.55, 2.8, 1.3), (0.32, 1.25, 2.8, 1.3), (1,), (1,), (0.32, 4.0)),
        size=(638, 332),
        help_text="pophelp('pop_newset')",
        button_size=(86, 31),
        content_margins=(32, 24, 32, 18),
        row_spacing=8,
        controls=(
            ControlSpec("text", prompt, font_weight="bold"),
            ControlSpec("text", "Name it:", enabled=True),
            ControlSpec("edit", tag="setname", value=dataset_name),
            ControlSpec(
                "pushbutton",
                "Edit description",
                tag="editdescription",
                callback=CallbackSpec(
                    "show_message",
                    params={
                        "button": "editdescription",
                        "title": "Edit description",
                        "message": "Dataset description editing is available from the command line using the comments option.",
                    },
                ),
            ),
            ControlSpec(
                "checkbox",
                "",
                tag="savenew",
                value=False,
                callback=CallbackSpec(
                    "toggle_enabled", params={"source": "savenew", "targets": ("savefile", "savebrowse")}
                ),
            ),
            ControlSpec("text", "Save it as file:", enabled=True),
            ControlSpec("edit", tag="savefile", value="", enabled=False),
            ControlSpec(
                "pushbutton",
                "Browse",
                tag="savebrowse",
                enabled=False,
                callback=CallbackSpec(
                    "select_file", params={"button": "savebrowse", "target": "savefile", "mode": "save"}
                ),
            ),
            ControlSpec("spacer"),
            ControlSpec("text", old_prompt, font_weight="bold"),
            ControlSpec(
                "checkbox", "Overwrite it in memory (set=yes; unset=create a new dataset)", tag="overwrite", value=False
            ),
        ),
    )


def _run_gui(
    EEG: dict[str, Any],
    CURRENTSET: int | list[int] | tuple[int, ...] | None,
    options: dict[str, Any],
    *,
    renderer: Any | None,
) -> dict[str, Any] | None:
    result = inputgui(
        pop_newset_dialog_spec(EEG, CURRENTSET, guistring=str(options.get("guistring") or "")),
        renderer=renderer,
    )
    if result is None:
        return None
    overwrite = _gui_choice_is_overwrite(result.get("overwrite"))
    gui_options = {
        "setname": str(result.get("setname") or "").strip(),
        "overwrite": "on" if overwrite else "off",
        "gui": "off",
    }
    if "comments" in result:
        gui_options["comments"] = str(result.get("comments") or "")
    if _is_on(result.get("savenew")):
        gui_options["savenew"] = str(result.get("savefile") or "on").strip() or "on"
    return gui_options


def _apply_dataset_metadata(
    EEG: dict[str, Any] | list[dict[str, Any]],
    options: dict[str, Any],
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(EEG, list):
        return [deepcopy(item) for item in EEG]
    output = deepcopy(EEG)
    if "setname" in options and options["setname"] not in {None, ""}:
        output["setname"] = str(options["setname"])
    if "comments" in options:
        output["comments"] = str(options["comments"])
    output["saved"] = "no"
    return output


def _store_index(
    CURRENTSET: int | list[int] | tuple[int, ...] | None,
    EEG: dict[str, Any] | list[dict[str, Any]],
    options: dict[str, Any],
) -> int | list[int] | None:
    if isinstance(EEG, list):
        return list(CURRENTSET) if isinstance(CURRENTSET, (list, tuple)) and len(CURRENTSET) == len(EEG) else None
    if _is_on(options.get("overwrite", False)):
        return _first_currentset(CURRENTSET) or None
    return None


def _first_currentset(CURRENTSET: int | list[int] | tuple[int, ...] | None) -> int | None:
    if isinstance(CURRENTSET, (list, tuple)):
        return int(CURRENTSET[0]) if CURRENTSET else None
    return int(CURRENTSET) if CURRENTSET else None


def _save_existing_datasets(ALLEEG: list[dict[str, Any]], CURRENTSET: Any) -> None:
    indices = list(CURRENTSET) if isinstance(CURRENTSET, (list, tuple)) else [CURRENTSET]
    for index in indices:
        if not index:
            continue
        dataset = ALLEEG[int(index) - 1]
        filename = _dataset_filename(dataset)
        if filename is None:
            raise ValueError("saveold requires the existing dataset to have filename and filepath")
        pop_saveset(dataset, filename)


def _save_new_dataset(EEG: dict[str, Any] | list[dict[str, Any]], target: Any) -> None:
    if isinstance(EEG, list):
        raise ValueError("savenew for multiple datasets requires explicit per-dataset paths")
    if isinstance(target, str) and target.lower() not in {"on", "yes", "true"}:
        filename = target
    else:
        filename = _dataset_filename(EEG)
    if filename is None:
        raise ValueError("savenew requires a filename or dataset filename/filepath metadata")
    pop_saveset(EEG, filename)
    EEG["saved"] = "yes"


def _dataset_filename(EEG: dict[str, Any]) -> str | None:
    filename = str(EEG.get("filename") or "")
    if not filename:
        return None
    filepath = str(EEG.get("filepath") or "")
    return str(Path(filepath) / filename) if filepath else filename


def _command_options(options: dict[str, Any], store_index: Any) -> dict[str, Any]:
    command_options = {}
    for key in ("setname", "comments", "overwrite", "saveold", "savenew", "study"):
        if key in options and _has_value(options[key]):
            command_options[key] = options[key]
    if store_index is not None and command_options.get("overwrite") == "off":
        command_options.pop("overwrite", None)
    return command_options


def _history_command(options: dict[str, Any]) -> str:
    if not options:
        return "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET);"
    parts = []
    for key, value in options.items():
        parts.extend([format_history_value(key), format_history_value(value, bool_style="onoff")])
    return f"[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, {', '.join(parts)});"


def _is_on(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "on", "true", "yes"}
    return bool(value)


def _should_save(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "off", "false", "no"}
    return bool(value)


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


def _gui_choice_is_overwrite(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return "overwrite" in value.lower()
    try:
        return int(value) == 2
    except (TypeError, ValueError):
        return False


def _currentset_label(CURRENTSET: Any) -> str:
    if isinstance(CURRENTSET, (list, tuple)):
        return ", ".join(str(item) for item in CURRENTSET) if CURRENTSET else "0"
    return str(CURRENTSET or 0)
