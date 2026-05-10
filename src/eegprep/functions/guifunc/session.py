"""State container for the EEGPrep EEGLAB-style GUI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

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
        self.CURRENTSET = list(stored_index) if isinstance(stored_index, list) else [int(stored_index)]
        if mark_saved:
            self.mark_current_saved()
        self.add_history(command)
        return stored_index

    def retrieve(self, indices: int | list[int]) -> dict[str, Any] | list[dict[str, Any]]:
        """Select dataset(s) from ALLEEG using 1-based indices."""
        eeg, self.ALLEEG, current = eeg_retrieve(self.ALLEEG, indices)
        self.EEG = eeg
        self.CURRENTSET = list(current) if isinstance(current, list) else [int(current)]
        return eeg

    def delete_current(self) -> None:
        """Delete the current dataset selection from memory."""
        if not self.CURRENTSET:
            return
        deleted_indices = list(self.CURRENTSET)
        self.ALLEEG, command = pop_delset(self.ALLEEG, self.CURRENTSET)
        self.add_history(command)
        if self.ALLEEG:
            self.retrieve(min(min(deleted_indices), len(self.ALLEEG)))
            return
        self.CURRENTSET = []
        self.EEG = eeg_emptyset()

    def clear_all(self) -> None:
        """Clear all datasets and study state."""
        self.EEG = eeg_emptyset()
        self.ALLEEG = []
        self.CURRENTSET = []
        self.STUDY = None
        self.CURRENTSTUDY = 0
        self.add_history("STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];")

    def add_history(self, command: str | None) -> None:
        """Append an EEGLAB-style command to session history."""
        self.LASTCOM = eegh(command, self.ALLCOM)

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
            if not has_eeg_data(dataset):
                continue
            setname = str(dataset.get("setname") or "(no dataset name)")
            summaries.append((index, f"Dataset {index}:{setname}", index in self.CURRENTSET))
        return summaries

    def clone_current(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Return a deep copy of the current EEG selection."""
        return deepcopy(self.EEG)


def _is_epoched(eeg: dict[str, Any]) -> bool:
    return int(eeg.get("trials", 1) or 1) > 1 or float(eeg.get("xmin", 0) or 0) != 0


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
