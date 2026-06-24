"""State container for the EEGPrep EEGLAB-style GUI."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import threading
from typing import Any, Callable, Iterator

import numpy as np

from eegprep.functions.adminfunc.eegh import eegh
from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
from eegprep.functions.adminfunc.eeg_store import eeg_store
from eegprep.functions.adminfunc.pop_delset import pop_delset
from eegprep.functions.adminfunc.storage import offload_storedisk_datasets
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset


_UNSET = object()


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
    """Normalize EEGLAB-facing 1-based dataset indices for session state.

    Empty selection is represented by ``None``, ``""``, scalar ``0``, an empty
    iterable, or a single-item iterable containing ``0`` when ``allow_empty`` is
    true. Mixed non-positive entries such as ``[0, 1]`` are invalid.
    """
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
    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    def add_change_listener(self, listener: Callable[["EEGPrepSession"], None]) -> None:
        """Register a callback that runs after session state changes."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_change_listener(self, listener: Callable[["EEGPrepSession"], None]) -> None:
        """Remove a previously registered session change callback."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def add_command_echo_listener(self, listener: Callable[[str], None]) -> None:
        """Register a callback for GUI commands to display in the console."""
        with self._lock:
            if listener not in self._command_echo_listeners:
                self._command_echo_listeners.append(listener)

    def remove_command_echo_listener(self, listener: Callable[[str], None]) -> None:
        """Remove a previously registered command echo callback."""
        with self._lock:
            if listener in self._command_echo_listeners:
                self._command_echo_listeners.remove(listener)

    def add_gui_action_listener(self, listener: Callable[[str, str], None]) -> None:
        """Register a callback for GUI action start/end notifications."""
        with self._lock:
            if listener not in self._gui_action_listeners:
                self._gui_action_listeners.append(listener)

    def remove_gui_action_listener(self, listener: Callable[[str, str], None]) -> None:
        """Remove a previously registered GUI action callback."""
        with self._lock:
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
        index: int | list[int] | None = None,
    ) -> int | list[int]:
        """Store ``eeg`` in ALLEEG and select it."""
        with self._lock:
            if index is not None:
                if new:
                    raise ValueError("new and index cannot both be set")
                normalized_index = normalize_dataset_indices(index, allow_empty=False)
                if isinstance(eeg, list):
                    if len(normalized_index) != len(eeg):
                        raise ValueError("Length of EEG list must equal length of index")
                    store_index: int | list[int] = normalized_index
                else:
                    if len(normalized_index) != 1:
                        raise ValueError("A single EEG dataset must be stored to one index")
                    store_index = normalized_index[0]
            elif isinstance(eeg, list):
                if new:
                    store_index = [0] * len(eeg)
                elif len(self.CURRENTSET) == len(eeg):
                    store_index = list(self.CURRENTSET)
                else:
                    store_index = None
            else:
                store_index = 0 if new or not self.CURRENTSET else self.CURRENTSET[0]
            self.ALLEEG, checked, stored_index = eeg_store(self.ALLEEG, eeg, store_index)
            self.EEG = checked
            self.CURRENTSET = normalize_dataset_indices(stored_index, allow_empty=False)
            self._append_current_dataset_history(command)
            if mark_saved:
                self.mark_current_saved()
            self.add_history(command, notify=False)
            self.notify_changed()
            return stored_index

    def retrieve(self, indices: int | list[int]) -> dict[str, Any] | list[dict[str, Any]]:
        """Select dataset(s) from ALLEEG using 1-based indices."""
        with self._lock:
            selection = normalize_dataset_indices(indices, allow_empty=False)
            use_vector = isinstance(indices, (list, tuple)) or (isinstance(indices, np.ndarray) and indices.ndim > 0)
            eeg, self.ALLEEG, current = eeg_retrieve(self.ALLEEG, selection if use_vector else selection[0])
            self.EEG = eeg
            self.CURRENTSET = normalize_dataset_indices(current, allow_empty=False)
            self.notify_changed()
            return eeg

    def apply_workspace_state(
        self,
        *,
        eeg: Any = _UNSET,
        alleeg: Any = _UNSET,
        currentset: Any = _UNSET,
        allcom: Any = _UNSET,
        lastcom: Any = _UNSET,
        study: Any = _UNSET,
        currentstudy: Any = _UNSET,
        command: str = "",
        append_dataset_history: bool = False,
    ) -> None:
        """Apply a GUI/console workspace update as one session transaction."""
        with self._lock:
            dataset_changed = eeg is not _UNSET or alleeg is not _UNSET or currentset is not _UNSET
            if dataset_changed:
                resolved_alleeg = self.ALLEEG if alleeg is _UNSET else alleeg
                if not isinstance(resolved_alleeg, list):
                    raise ValueError("ALLEEG must be a list of EEG datasets")
                resolved_currentset = (
                    list(self.CURRENTSET)
                    if currentset is _UNSET
                    else normalize_dataset_indices(currentset, allow_empty=True)
                )
                if resolved_currentset and max(resolved_currentset) > len(resolved_alleeg):
                    raise ValueError("CURRENTSET contains indices outside ALLEEG")
                resolved_eeg = self._resolve_workspace_eeg(eeg, resolved_alleeg, resolved_currentset)
                current = resolved_eeg if isinstance(resolved_eeg, list) else [resolved_eeg]
                if resolved_currentset and len(current) != len(resolved_currentset):
                    raise ValueError("EEG selection length must match CURRENTSET")
                self.ALLEEG = resolved_alleeg
                self.EEG = resolved_eeg
                self.CURRENTSET = resolved_currentset
                self._mirror_current_eeg_into_alleeg()
                if append_dataset_history:
                    self._append_current_dataset_history(command)
                offload_storedisk_datasets(self.ALLEEG, set(self.CURRENTSET))

            if allcom is not _UNSET:
                if not isinstance(allcom, list):
                    raise ValueError("ALLCOM must be a list of command strings")
                self.ALLCOM = [str(item) for item in allcom if str(item).strip()]
                self.LASTCOM = self.ALLCOM[-1] if self.ALLCOM else ""
            if lastcom is not _UNSET:
                last_command = str(lastcom or "").strip()
                if last_command and (not self.ALLCOM or self.ALLCOM[-1] != last_command):
                    self.ALLCOM.append(last_command)
                self.LASTCOM = last_command

            if study is not _UNSET:
                self.STUDY = study
                if currentstudy is _UNSET:
                    self.CURRENTSTUDY = 1 if study else 0
            if currentstudy is not _UNSET:
                self.CURRENTSTUDY = int(currentstudy or 0)

            self.add_history(command, notify=False)
            self.notify_changed()

    def delete_current(self) -> None:
        """Delete the current dataset selection from memory."""
        with self._lock:
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
        with self._lock:
            self.EEG = eeg_emptyset()
            self.ALLEEG = []
            self.CURRENTSET = []
            self.STUDY = None
            self.CURRENTSTUDY = 0
            self.add_history("STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];")

    def set_study(
        self,
        study: dict[str, Any] | None,
        alleeg: list[dict[str, Any]] | None = None,
        *,
        command: str = "",
    ) -> None:
        """Set STUDY/CURRENTSTUDY and optionally replace loaded datasets."""
        with self._lock:
            self.STUDY = study
            self.CURRENTSTUDY = 1 if study else 0
            if alleeg is not None:
                self.ALLEEG = alleeg
                if self.ALLEEG and (not self.CURRENTSET or max(self.CURRENTSET) > len(self.ALLEEG)):
                    self.CURRENTSET = [1]
                    self.EEG = self.ALLEEG[0]
                elif self.ALLEEG and self.CURRENTSET:
                    selected = [self.ALLEEG[index - 1] for index in self.CURRENTSET]
                    self.EEG = selected if len(selected) > 1 else selected[0]
                elif not self.ALLEEG:
                    self.CURRENTSET = []
                    self.EEG = eeg_emptyset()
                offload_storedisk_datasets(self.ALLEEG, set(self.CURRENTSET))
            self.add_history(command, notify=False)
            self.notify_changed()

    def _resolve_workspace_eeg(
        self,
        eeg: Any,
        alleeg: list[dict[str, Any]],
        currentset: list[int],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if eeg is not _UNSET:
            return eeg
        if not currentset:
            return eeg_emptyset()
        selected = [alleeg[index - 1] for index in currentset]
        return selected if len(selected) > 1 else selected[0]

    def _mirror_current_eeg_into_alleeg(self) -> None:
        if not self.CURRENTSET:
            return
        current = self.EEG if isinstance(self.EEG, list) else [self.EEG]
        if len(current) != len(self.CURRENTSET):
            raise ValueError("EEG selection length must match CURRENTSET")
        for index, eeg in zip(self.CURRENTSET, current):
            if 1 <= index <= len(self.ALLEEG):
                self.ALLEEG[index - 1] = eeg

    def select_study(self, *, command: str = "CURRENTSTUDY = 1") -> None:
        """Select the current STUDY set in the shared workspace."""
        with self._lock:
            if not self.STUDY:
                raise ValueError("No current STUDY")
            self.CURRENTSTUDY = 1
            self.add_history(command)

    def add_history(self, command: str | None, *, notify: bool = True) -> None:
        """Append an EEGLAB-style command to session history."""
        with self._lock:
            if not command:
                if notify:
                    self.notify_changed()
                return
            self.LASTCOM = eegh(command, self.ALLCOM)
            if notify:
                self.notify_changed()

    def clear_history(self, *, notify: bool = True) -> None:
        """Clear command history and LASTCOM as one session mutation."""
        with self._lock:
            self.ALLCOM.clear()
            self.LASTCOM = ""
            if notify:
                self.notify_changed()

    def remove_history(self, count: int, *, notify: bool = True) -> None:
        """Remove the most recent ``count`` command-history entries."""
        with self._lock:
            remove_count = min(max(int(count), 0), len(self.ALLCOM))
            if remove_count:
                del self.ALLCOM[-remove_count:]
            self.LASTCOM = self.ALLCOM[-1] if self.ALLCOM else ""
            if notify:
                self.notify_changed()

    def history_command_at(self, index: int) -> str:
        """Return the 1-based command from most recent history first."""
        if index < 1 or index > len(self.ALLCOM):
            return ""
        return list(reversed(self.ALLCOM))[index - 1]

    def clear_last_command(self, *, notify: bool = True) -> None:
        """Clear LASTCOM without deleting ALLCOM."""
        with self._lock:
            self.LASTCOM = ""
            if notify:
                self.notify_changed()

    def _append_current_dataset_history(self, command: str | None) -> None:
        with self._lock:
            if not command:
                return
            current = self.EEG if isinstance(self.EEG, list) else [self.EEG]
            for eeg in current:
                if isinstance(eeg, dict):
                    eegh(command, eeg)

    def mark_current_saved(self) -> None:
        """Mark the current dataset selection as saved in EEG and ALLEEG."""
        with self._lock:
            current = self.EEG if isinstance(self.EEG, list) else [self.EEG]
            for index, eeg in zip(self.CURRENTSET, current):
                eeg["saved"] = "yes"
                if 1 <= index <= len(self.ALLEEG):
                    self.ALLEEG[index - 1]["saved"] = "yes"
            offload_storedisk_datasets(self.ALLEEG, set(self.CURRENTSET))

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
