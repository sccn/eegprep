"""Execute a registered EEGPrep menu workflow without opening its dialog."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePath
from typing import Any

from eegprep.extension_runtime import ExtensionRuntime
from eegprep.functions.guifunc.eeglab_menu import eeglab_menus
from eegprep.functions.guifunc.menu_spec import MenuItemSpec
from eegprep.functions.guifunc.session import EEGPrepSession, has_eeg_data
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_chanedit import pop_chanedit
from eegprep.functions.popfunc.pop_comments import pop_comments
from eegprep.functions.popfunc.pop_editeventvals import pop_editeventvals
from eegprep.functions.popfunc.pop_editset import pop_editset
from eegprep.functions.popfunc.pop_importdata import pop_importdata
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_resample import pop_resample
from eegprep.functions.popfunc.pop_saveset import pop_saveset


_Parameters = Sequence[Any] | Mapping[str, Any] | None
_CURRENT_DATASET_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "pop_chanedit": pop_chanedit,
    "pop_editeventvals": pop_editeventvals,
    "pop_editset": pop_editset,
    "pop_resample": pop_resample,
}
_SUPPORTED_FUNCTIONS = {*_CURRENT_DATASET_FUNCTIONS, "pop_comments", "pop_importdata", "pop_loadset", "pop_saveset"}
_CONTROLLED_KEYWORDS = {"gui", "renderer", "return_com"}


def eeglab_execmenu(
    label: str,
    function: str,
    parameters: _Parameters,
    *,
    session: EEGPrepSession,
    extension_runtime: ExtensionRuntime | None = None,
) -> str:
    """Run a menu-bound ``pop_*`` workflow with explicit parameters.

    The menu label and function name must identify the same registered leaf in
    EEGPrep's declarative menu tree. Only explicitly supported, noninteractive
    workflows are callable: this function never evaluates callback text or
    arbitrary history strings.

    Args:
        label: Exact visible menu label, such as ``"Change sampling rate"``.
        function: Public function name present in that menu action.
        parameters: Positional arguments, EEGLAB-style key/value pairs, or a
            mapping of keyword arguments to append to the menu workflow.
        session: Shared session to read and update.
        extension_runtime: Optional extension registry used while resolving the
            current menu tree.

    Returns:
        The replayable command appended to ``session.ALLCOM``.

    Raises:
        ValueError: If the label and function do not identify one menu action.
        NotImplementedError: If the resolved action has no safe noninteractive
            dispatcher.
    """
    action = _resolve_menu_action(label, function, extension_runtime=extension_runtime)
    if function not in _SUPPORTED_FUNCTIONS:
        raise NotImplementedError(
            f"Menu action {action!r} does not support parameterized execution through eeglab_execmenu"
        )
    args, kwargs = _normalize_parameters(parameters)
    if function == "pop_importdata":
        _validate_import_target(session)
        eeg_out, command = pop_importdata(*args, **kwargs, return_com=True)
        _store_imported_dataset(session, eeg_out, command)
        return command
    if function == "pop_loadset":
        _validate_import_target(session)
        eeg_out = pop_loadset(*args, **kwargs)
        command = _history_command(function, args, kwargs)
        _store_imported_dataset(session, eeg_out, command)
        return command

    eeg = _require_current_dataset(session)
    if function == "pop_saveset":
        args, kwargs = _apply_action_defaults(action, args, kwargs)
        eeg_out = pop_saveset(eeg, *args, **kwargs)
        command = _history_command(function, args, kwargs, include_eeg=True)
        _store_current_dataset(session, eeg_out, command, mark_saved=True)
        return command
    if function == "pop_comments":
        eeg_out, command = pop_comments(
            eeg,
            "About this dataset",
            *args,
            **kwargs,
            gui=False,
            return_com=True,
        )
    else:
        target = _CURRENT_DATASET_FUNCTIONS[function]
        eeg_out, command = target(eeg, *args, **kwargs, gui=False, return_com=True)
    _store_current_dataset(session, eeg_out, command)
    return command


def _resolve_menu_action(
    label: str,
    function: str,
    *,
    extension_runtime: ExtensionRuntime | None,
) -> str:
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty menu label")
    if not isinstance(function, str) or not function:
        raise ValueError("function must be a non-empty function name")
    items = eeglab_menus(
        all_menus=True,
        include_plugins=extension_runtime is not None,
        extension_runtime=extension_runtime,
    )
    labelled = [item for item in _walk_menu_items(items) if item.label == label]
    matches = [item for item in labelled if item.action and item.action.partition(":")[0] == function]
    if len(matches) == 1:
        return str(matches[0].action)
    if len(matches) > 1:
        raise ValueError(f"Menu label {label!r} and function {function!r} are ambiguous")
    if not labelled:
        raise ValueError(f"Could not find menu with label {label!r}")
    actions = sorted({item.action for item in labelled if item.action})
    if not actions:
        raise ValueError(f"Menu label {label!r} is not an executable menu item")
    raise ValueError(f"Menu label {label!r} is registered for {', '.join(actions)}, not {function!r}")


def _walk_menu_items(items: tuple[MenuItemSpec, ...]) -> list[MenuItemSpec]:
    walked: list[MenuItemSpec] = []
    for item in items:
        walked.append(item)
        walked.extend(_walk_menu_items(item.children))
    return walked


def _normalize_parameters(parameters: _Parameters) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if parameters is None:
        return (), {}
    if isinstance(parameters, Mapping):
        kwargs = {str(key): value for key, value in parameters.items()}
        controlled = sorted(_CONTROLLED_KEYWORDS & set(kwargs))
        if controlled:
            raise ValueError(f"eeglab_execmenu controls parameter(s): {', '.join(controlled)}")
        return (), kwargs
    if isinstance(parameters, Sequence) and not isinstance(parameters, (str, bytes, bytearray)):
        return tuple(parameters), {}
    raise TypeError("parameters must be a sequence, mapping, or None")


def _apply_action_defaults(
    action: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    _base, _separator, variant = action.partition(":")
    if variant != "resave" or _has_option(args, kwargs, "savemode"):
        return args, kwargs
    return (*args, "savemode", "resave"), kwargs


def _has_option(args: tuple[Any, ...], kwargs: Mapping[str, Any], option: str) -> bool:
    if any(str(key).lower() == option for key in kwargs):
        return True
    return any(isinstance(value, str) and value.lower() == option for value in args[::2])


def _require_current_dataset(session: EEGPrepSession) -> dict[str, Any]:
    if len(session.CURRENTSET) != 1:
        raise ValueError("eeglab_execmenu requires exactly one current dataset")
    eeg = session.current_eeg()
    if isinstance(eeg, list):
        if len(eeg) != 1:
            raise ValueError("eeglab_execmenu requires exactly one current dataset")
        eeg = eeg[0]
    if not has_eeg_data(eeg):
        raise ValueError("No current dataset")
    return eeg


def _validate_import_target(session: EEGPrepSession) -> None:
    if len(session.CURRENTSET) > 1:
        raise ValueError("eeglab_execmenu requires at most one current dataset for an import")


def _store_imported_dataset(session: EEGPrepSession, eeg: dict[str, Any], command: str) -> None:
    session.echo_command(command)
    if session.CURRENTSET:
        session.store_current(eeg, index=session.CURRENTSET[0], command=command)
        return
    session.store_current(eeg, new=True, command=command)


def _store_current_dataset(
    session: EEGPrepSession,
    eeg: dict[str, Any],
    command: str,
    *,
    mark_saved: bool = False,
) -> None:
    session.echo_command(command)
    session.store_current(eeg, index=session.CURRENTSET[0], command=command, mark_saved=mark_saved)


def _history_command(
    function: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    include_eeg: bool = False,
) -> str:
    values = ["EEG"] if include_eeg else []
    values.extend(_history_value(value) for value in args)
    for key, value in kwargs.items():
        values.extend((format_history_value(str(key)), _history_value(value)))
    return f"EEG = {function}({', '.join(values)});"


def _history_value(value: Any) -> str:
    if isinstance(value, PurePath):
        value = value.as_posix()
    return format_history_value(value, cell_for_sequence="any_strings")


__all__ = ["eeglab_execmenu"]
