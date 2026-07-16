"""Shared policy helpers for Qt file dialogs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


# Note: This context scopes overrides only for synchronous modal dialogs (exec()).
# Non-modal (show()) dialogs spawned outside the dispatch frame fall back to EEG_OPTIONS.
_NATIVE_FILE_DIALOG_OVERRIDE: ContextVar[bool | None] = ContextVar(
    "eegprep_native_file_dialog_override",
    default=None,
)


@contextmanager
def native_file_dialog_override(value: bool | None) -> Iterator[None]:
    """Apply an explicit file-dialog policy during one GUI dispatch."""
    if value is None:
        yield
        return
    token = _NATIVE_FILE_DIALOG_OVERRIDE.set(bool(value))
    try:
        yield
    finally:
        _NATIVE_FILE_DIALOG_OVERRIDE.reset(token)


def file_dialog_kwargs(
    qt_widgets: Any,
    *,
    native_file_dialogs: bool | None = None,
    directories: bool = False,
) -> dict[str, Any]:
    """Return Qt file-dialog flags after resolving explicit and global policy.

    An explicit function argument wins, followed by the current GUI dispatch
    override and then ``EEG_OPTIONS["option_native_dialogs"]``.
    """
    if _use_native_file_dialogs(native_file_dialogs):
        return {}

    options = _qt_enum_value(qt_widgets.QFileDialog, "Option", "DontUseNativeDialog")
    if directories:
        show_dirs = _qt_enum_value(qt_widgets.QFileDialog, "Option", "ShowDirsOnly")
        if options is None:
            options = show_dirs
        elif show_dirs is not None:
            options = options | show_dirs
    return {"options": options} if options is not None else {}


def _use_native_file_dialogs(explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    scoped = _NATIVE_FILE_DIALOG_OVERRIDE.get()
    if scoped is not None:
        return scoped
    from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS

    return bool(int(EEG_OPTIONS.get("option_native_dialogs", 0) or 0))


def _qt_enum_value(owner: Any, enum_name: str, value_name: str) -> Any | None:
    enum = getattr(owner, enum_name, None)
    value = getattr(enum, value_name, None) if enum is not None else None
    if value is not None:
        return value
    return getattr(owner, value_name, None)
