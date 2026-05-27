"""Edit EEG dataset comments with EEGLAB ``pop_comments`` semantics."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value


def pop_comments(
    EEG: dict[str, Any] | str | list[str] | tuple[str, ...] | np.ndarray,
    plottitle: str = "",
    newcomments: Any = None,
    concat: bool | int = False,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
) -> Any:
    """Edit or assign dataset comments.

    Passing an EEG dictionary updates ``EEG["comments"]`` and returns a copied
    EEG dictionary. Passing a string/list returns the edited comment text.
    """
    is_eeg = isinstance(EEG, dict)
    original = copy.copy(EEG) if is_eeg else EEG
    old_text = _comments_to_text(EEG.get("comments", "") if is_eeg else EEG)
    if gui is None:
        gui = newcomments is None

    if gui:
        result = inputgui(pop_comments_dialog_spec(plottitle, old_text), renderer=renderer)
        if result is None:
            return (original, "") if return_com else original
        new_text = _comments_to_text(result.get("comments", ""))
    else:
        if newcomments is None:
            new_text = old_text
        else:
            new_text = _comments_to_text(newcomments)

    command_text = new_text
    concat_enabled = _truthy_concat(concat)
    if concat_enabled:
        new_text = _join_comments(old_text, new_text)

    command = _history_command(command_text, concat=concat_enabled, is_eeg=is_eeg) if return_com else ""
    if is_eeg:
        output = copy.deepcopy(EEG)
        output["comments"] = new_text
    else:
        output = new_text
    return (output, command) if return_com else output


def pop_comments_dialog_spec(title: str = "", comments: str = "") -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_comments``."""
    display_title = title or "Read/Enter text -- pop_comments()"
    controls = []
    if title:
        controls.append(ControlSpec("text", title, font_weight="bold"))
    controls.append(ControlSpec("textarea", tag="comments", value=comments))
    return DialogSpec(
        title="Read/Enter text -- pop_comments()",
        function_name="pop_comments",
        eeglab_source="functions/popfunc/pop_comments.m",
        size=(1394, 840),
        geometry=tuple(1 for _ in controls),
        content_margins=(170, 35, 170, 50),
        row_spacing=18,
        geomvert=(0.05, 1.0),
        help_text="pop_comments",
        show_help_button=False,
        ok_label="SAVE",
        cancel_label="CANCEL",
        button_size=(150, 45),
        cancel_first=True,
        controls=tuple(controls),
        known_differences=("EEGPrep renders the editable comment area with Qt QTextEdit.", display_title),
    )


def _comments_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value.rstrip()
    if isinstance(value, (list, tuple)):
        return "\n".join(_comments_to_text(item) for item in value).rstrip()
    return str(value).rstrip()


def _join_comments(old_text: str, new_text: str) -> str:
    if not old_text:
        return new_text
    if not new_text:
        return old_text
    return f"{old_text.rstrip()}\n{new_text.lstrip()}".rstrip()


def _truthy_concat(value: bool | int) -> bool:
    if isinstance(value, str):
        raise TypeError("concat must be 0 or 1")
    return bool(value)


def _history_command(newcomments: str, *, concat: bool, is_eeg: bool) -> str:
    target = "EEG" if is_eeg else "comments"
    command = f"{target} = pop_comments({target}, '', {_history_comments_value(newcomments)}"
    if concat:
        command += ", 1"
    return command + ");"


def _history_comments_value(comments: str) -> str:
    if "\n" not in comments:
        return format_history_value(comments)
    lines = [line if line else " " for line in comments.split("\n")]
    return format_history_value(lines, cell_for_sequence="always")


__all__ = ["pop_comments", "pop_comments_dialog_spec"]
