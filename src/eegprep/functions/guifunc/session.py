"""State container for the EEGPrep EEGLAB-style GUI."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

from eegprep.functions.adminfunc.eegh import eegh
from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
from eegprep.functions.adminfunc.eeg_store import eeg_store
from eegprep.functions.adminfunc.pop_delset import pop_delset
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset


def has_eeg_data(eeg: Any) -> bool:
    """Return whether an EEG-like object contains non-empty data."""
    if not isinstance(eeg, dict):
        return False
    data = eeg.get("data")
    if data is None:
        return False
    if isinstance(data, np.ndarray):
        return data.size > 0
    if isinstance(data, list):
        return len(data) > 0
    return True


def normalize_dataset_indices(indices: Any, *, allow_empty: bool = True) -> list[int]:
    """Normalize EEGLAB-facing 1-based dataset indices for session state."""
    if indices is None or (isinstance(indices, str) and indices == ""):
        if allow_empty:
            return []
        raise ValueError("Dataset selection must contain at least one index")
    if isinstance(indices, bool):
        raise ValueError("Dataset indices must be 1-based integers")
    if isinstance(indices, np.ndarray):
        values = [indices.item()] if indices.ndim == 0 else indices.ravel().tolist()
    elif isinstance(indices, (int, np.integer)):
        values = [int(indices)]
    elif isinstance(indices, (float, np.floating)):
        if not float(indices).is_integer():
            raise ValueError("Dataset indices must be integers")
        values = [int(indices)]
    elif isinstance(indices, Iterable) and not isinstance(indices, (str, bytes, dict)):
        values = list(indices)
    else:
        raise ValueError("Dataset selection must be a 1-based integer or iterable of integers")

    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            raise ValueError("Dataset indices must be 1-based integers")
        if isinstance(value, (float, np.floating)) and not float(value).is_integer():
            raise ValueError("Dataset indices must be integers")
        index = int(value)
        if index < 1:
            if allow_empty and len(values) == 1 and index == 0:
                continue
            raise ValueError("EEGLAB dataset indices are 1-based")
        if index in seen:
            raise ValueError("Dataset selection cannot contain duplicate indices")
        seen.add(index)
        normalized.append(index)
    if not normalized and not allow_empty:
        raise ValueError("Dataset selection must contain at least one index")
    return normalized


@dataclass
class EEGPrepSession:
    """EEGLAB-like GUI state without module globals."""

    EEG: dict[str, Any] | list[dict[str, Any]] = field(default_factory=eeg_emptyset)
    ALLEEG: list[dict[str, Any]] = field(default_factory=list)
    CURRENTSET: list[int] = field(default_factory=list)
    ALLCOM: list[str] = field(default_factory=list)
    LASTCOM: str = ""
    STUDY: dict[str, Any] | None = None
    CURRENTSTUDY: int = 0
    PLUGINLIST: list[dict[str, Any]] = field(default_factory=list)
    _listeners: list[Callable[["EEGPrepSession"], None]] = field(default_factory=list, init=False, repr=False)
    _command_echo_listeners: list[Callable[[str], None]] = field(default_factory=list, init=False, repr=False)
    _gui_action_listeners: list[Callable[[str, str], None]] = field(default_factory=list, init=False, repr=False)

    def add_change_listener(self, listener: Callable[["EEGPrepSession"], None]) -> None:
        """Register a callback that runs after session state changes."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_change_listener(self, listener: Callable[["EEGPrepSession"], None]) -> None:
        """Remove a previously registered session change callback."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def add_command_echo_listener(self, listener: Callable[[str], None]) -> None:
        """Register a callback for GUI commands to display in the console."""
        if listener not in self._command_echo_listeners:
            self._command_echo_listeners.append(listener)

    def remove_command_echo_listener(self, listener: Callable[[str], None]) -> None:
        """Remove a previously registered command echo callback."""
        if listener in self._command_echo_listeners:
            self._command_echo_listeners.remove(listener)

    def add_gui_action_listener(self, listener: Callable[[str, str], None]) -> None:
        """Register a callback for GUI action start/end notifications."""
        if listener not in self._gui_action_listeners:
            self._gui_action_listeners.append(listener)

    def remove_gui_action_listener(self, listener: Callable[[str, str], None]) -> None:
        """Remove a previously registered GUI action callback."""
        if listener in self._gui_action_listeners:
            self._gui_action_listeners.remove(listener)

    def begin_gui_action(self, action: str) -> None:
        """Notify listeners that a GUI action is about to run."""
        for listener in list(self._gui_action_listeners):
            listener("begin", action)

    def end_gui_action(self, action: str) -> None:
        """Notify listeners that a GUI action has finished."""
        for listener in list(self._gui_action_listeners):
            listener("end", action)

    @contextmanager
    def gui_action(self, action: str) -> Iterator[None]:
        """Wrap a user-triggered GUI action for console/output synchronization."""
        self.begin_gui_action(action)
        try:
            yield
        finally:
            self.end_gui_action(action)

    def echo_command(self, command: str | None) -> None:
        """Display a GUI command without mutating session history."""
        if not command:
            return
        for listener in list(self._command_echo_listeners):
            listener(command)

    def notify_changed(self) -> None:
        """Notify listeners that session-backed state changed."""
        for listener in list(self._listeners):
            listener(self)

    def current_eeg(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Return the current EEG selection."""
        return self.EEG

    def current_set_value(self) -> int | list[int]:
        """Return EEGLAB-style CURRENTSET scalar/list value."""
        if not self.CURRENTSET:
            return 0
        if len(self.CURRENTSET) == 1:
            return self.CURRENTSET[0]
        return list(self.CURRENTSET)

    def selected_dataset_indices(self) -> list[int]:
        """Return the selected EEGLAB-facing dataset indices in order."""
        return list(self.CURRENTSET)

    def store_current(
        self,
        eeg: dict[str, Any] | list[dict[str, Any]],
        *,
        new: bool = False,
        command: str = "",
        mark_saved: bool = False,
    ) -> int | list[int]:
        """Store ``eeg`` in ALLEEG and select it."""
        if isinstance(eeg, list):
            if new:
                index: int | list[int] | None = [0] * len(eeg)
            elif len(self.CURRENTSET) == len(eeg):
                index = list(self.CURRENTSET)
            else:
                index = None
        else:
            index = 0 if new or not self.CURRENTSET else self.CURRENTSET[0]
        self.ALLEEG, checked, stored_index = eeg_store(self.ALLEEG, eeg, index)
        self.EEG = checked
        self.CURRENTSET = normalize_dataset_indices(stored_index, allow_empty=False)
        if mark_saved:
            self.mark_current_saved()
        self.add_history(command, notify=False)
        self.notify_changed()
        return stored_index

    def retrieve(self, indices: int | list[int]) -> dict[str, Any] | list[dict[str, Any]]:
        """Select dataset(s) from ALLEEG using 1-based indices."""
        selection = normalize_dataset_indices(indices, allow_empty=False)
        use_vector = isinstance(indices, (list, tuple)) or (isinstance(indices, np.ndarray) and indices.ndim > 0)
        eeg, self.ALLEEG, current = eeg_retrieve(self.ALLEEG, selection if use_vector else selection[0])
        self.EEG = eeg
        self.CURRENTSET = normalize_dataset_indices(current, allow_empty=False)
        self.notify_changed()
        return eeg

    def delete_current(self) -> None:
        """Delete the current dataset selection from memory."""
        if not self.CURRENTSET:
            return
        deleted_indices = list(self.CURRENTSET)
        self.ALLEEG, command = pop_delset(self.ALLEEG, self.CURRENTSET)
        self.add_history(command, notify=False)
        if self.ALLEEG:
            self.retrieve(min(min(deleted_indices), len(self.ALLEEG)))
            return
        self.CURRENTSET = []
        self.EEG = eeg_emptyset()
        self.notify_changed()

    def clear_all(self) -> None:
        """Clear all datasets and study state."""
        self.EEG = eeg_emptyset()
        self.ALLEEG = []
        self.CURRENTSET = []
        self.STUDY = None
        self.CURRENTSTUDY = 0
        self.add_history("STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];")

    def add_history(self, command: str | None, *, notify: bool = True) -> None:
        """Append an EEGLAB-style command to session history."""
        self.LASTCOM = eegh(command, self.ALLCOM)
        if notify:
            self.notify_changed()

    def mark_current_saved(self) -> None:
        """Mark the current dataset selection as saved in EEG and ALLEEG."""
        current = self.EEG if isinstance(self.EEG, list) else [self.EEG]
        for index, eeg in zip(self.CURRENTSET, current):
            eeg["saved"] = "yes"
            if 1 <= index <= len(self.ALLEEG):
                self.ALLEEG[index - 1]["saved"] = "yes"

    def menu_statuses(self) -> set[str]:
        """Return EEGLAB-style menu status tokens for the current state."""
        if self.CURRENTSTUDY == 1 and self.STUDY:
            return {"study"}
        eeg = self.EEG
        if isinstance(eeg, list) and len(eeg) > 1:
            return {"multiple_datasets"}
        if isinstance(eeg, list):
            eeg = eeg[0] if eeg else eeg_emptyset()
        if not has_eeg_data(eeg):
            return {"startup"}

        statuses = {"epoched_dataset"} if _is_epoched(eeg) else {"continuous_dataset"}
        if _chanloc_absent(eeg):
            statuses.add("chanloc_absent")
        if _ica_absent(eeg):
            statuses.add("ica_absent")
        if _roi_connect(eeg):
            statuses.add("roi_connect")
        return statuses

    def dataset_summaries(self) -> list[tuple[int, str, bool]]:
        """Return ``(index, label, selected)`` tuples for the Datasets menu."""
        summaries = []
        for index, dataset in enumerate(self.ALLEEG, start=1):
            if not isinstance(dataset, dict) or not dataset:
                continue
            setname = str(dataset.get("setname") or "(no dataset name)")
            summaries.append((index, f"Dataset {index}:{setname}", index in self.CURRENTSET))
        return summaries

    def clone_current(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Return a deep copy of the current EEG selection."""
        return deepcopy(self.EEG)


def _is_epoched(eeg: dict[str, Any]) -> bool:
    return int(eeg.get("trials", 1) or 1) > 1


def _chanloc_absent(eeg: dict[str, Any]) -> bool:
    chanlocs = eeg.get("chanlocs")
    if chanlocs is None:
        return True
    if isinstance(chanlocs, np.ndarray):
        if chanlocs.size == 0:
            return True
        chanlocs = chanlocs.tolist()
    if not chanlocs:
        return True
    first = chanlocs[0] if isinstance(chanlocs, list) else chanlocs
    return not isinstance(first, dict) or "theta" not in first


def _ica_absent(eeg: dict[str, Any]) -> bool:
    weights = eeg.get("icaweights")
    if weights is None:
        return True
    if isinstance(weights, np.ndarray):
        return weights.size == 0
    return not bool(weights)


def _roi_connect(eeg: dict[str, Any]) -> bool:
    roi = eeg.get("roi")
    return isinstance(roi, dict) and bool(roi.get("eeglab_using_roi"))
