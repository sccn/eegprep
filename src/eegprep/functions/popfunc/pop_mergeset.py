"""Merge EEGLAB-style EEG datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.session import normalize_dataset_indices
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._event_utils import events_as_list
from eegprep.functions.popfunc._pop_utils import format_history_value


def pop_mergeset(
    INEEG1: dict[str, Any] | list[dict[str, Any]],
    INEEG2: Any = None,
    keepall: int | bool = 0,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Merge two or more EEG datasets.

    When ``INEEG1`` is an ``ALLEEG`` list, ``INEEG2`` is interpreted as
    EEGLAB-facing 1-based dataset indices. ``keepall`` preserves ICA weights
    from the first dataset when possible, matching EEGLAB's option.
    """
    if gui is None:
        gui = INEEG2 is None and isinstance(INEEG1, list)
    indices: list[int] | None = None
    if gui:
        result = _run_gui(INEEG1 if isinstance(INEEG1, list) else [INEEG1], renderer=renderer)
        if result is None:
            return (INEEG1, "") if return_com else INEEG1
        INEEG2 = result["indices"]
        keepall = result["keepall"]
    if isinstance(INEEG1, list):
        if INEEG2 is None:
            raise ValueError("needs at least two datasets")
        indices = normalize_dataset_indices(INEEG2, allow_empty=False)
        if len(indices) < 2:
            raise ValueError("needs at least two datasets")
        datasets = [deepcopy(eeg_retrieve(INEEG1, index)[0]) for index in indices]
        merged = datasets[0]
        for dataset in datasets[1:]:
            merged = _merge_two(merged, dataset, bool(keepall))
    else:
        if INEEG2 is None:
            raise ValueError("needs at least two datasets")
        merged = _merge_two(INEEG1, INEEG2, bool(keepall))
    command = _history_command(indices, keepall)
    return (merged, command) if return_com else merged


def pop_mergeset_dialog_spec(ALLEEG: list[dict[str, Any]]) -> DialogSpec:
    """Return the EEGLAB-like ``pop_mergeset`` dialog spec."""
    default_indices = " ".join(str(index) for index in range(1, min(len(ALLEEG), 2) + 1)) or "1"
    return DialogSpec(
        title="Merge datasets -- pop_mergeset()",
        function_name="pop_mergeset",
        eeglab_source="functions/popfunc/pop_mergeset.m",
        size=(445, 165),
        content_margins=(42, 30, 42, 30),
        row_spacing=8,
        help_text="pophelp('pop_mergeset')",
        geometry=((3, 1), (3, 1)),
        controls=(
            ControlSpec("text", "Dataset indices to merge"),
            ControlSpec("edit", tag="indices", value=default_indices),
            ControlSpec("text", "Preserve ICA weights of the first dataset ?"),
            ControlSpec("checkbox", tag="keepall", value=False),
        ),
    )


def _run_gui(ALLEEG: list[dict[str, Any]], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_mergeset_dialog_spec(ALLEEG), renderer=renderer)
    if result is None:
        return None
    return {
        "indices": [int(float(token)) for token in str(result.get("indices") or "").strip("[]").split() if token],
        "keepall": int(bool(result.get("keepall"))),
    }


def _merge_two(first: dict[str, Any], second: dict[str, Any], keepall: bool) -> dict[str, Any]:
    if int(first.get("nbchan", 0)) != int(second.get("nbchan", 0)):
        raise ValueError("The two datasets must have the same number of channels")
    if float(first.get("srate", 0)) != float(second.get("srate", 0)):
        raise ValueError("The two datasets must have the same sampling rate")
    first_trials = int(first.get("trials", 1) or 1)
    second_trials = int(second.get("trials", 1) or 1)
    first_pnts = int(first.get("pnts", 0) or 0)
    second_pnts = int(second.get("pnts", 0) or 0)
    if (first_trials > 1 or second_trials > 1) and first_pnts != second_pnts:
        raise ValueError("The two epoched datasets must have the same number of points")

    output = deepcopy(first)
    other = deepcopy(second)
    first_data = np.asarray(output["data"])
    second_data = np.asarray(other["data"])
    if first_trials > 1 or second_trials > 1:
        output["data"] = np.concatenate(
            (_as_epoched(first_data, first_pnts), _as_epoched(second_data, second_pnts)), axis=2
        )
        output["trials"] = first_trials + second_trials
        output["pnts"] = first_pnts
        output["epoch"] = _merge_epochs(output.get("epoch"), other.get("epoch"), first_trials, second_trials)
    else:
        output["data"] = np.concatenate((first_data, second_data), axis=1)
        output["pnts"] = first_pnts + second_pnts
        output["trials"] = 1
    output["setname"] = "Merged datasets"
    output["saved"] = "no"
    output["event"] = _merge_events(output, other, first_pnts, first_trials, first_trials > 1 or second_trials > 1)
    output["urevent"] = _merge_urevents(output, other, first_pnts, first_trials, first_trials > 1 or second_trials > 1)
    output["reject"] = {}
    output["specicaact"] = []
    output["specdata"] = []
    output["icaact"] = np.array([])
    if not keepall:
        output["icawinv"] = np.array([])
        output["icasphere"] = np.array([])
        output["icaweights"] = np.array([])
        output["stats"] = {}
    return eeg_checkset(output, "eventconsistency")


def _as_epoched(data: np.ndarray, pnts: int) -> np.ndarray:
    if data.ndim == 3:
        return data
    return data.reshape(data.shape[0], pnts, 1)


def _merge_events(
    first: dict[str, Any],
    second: dict[str, Any],
    first_pnts: int,
    first_trials: int,
    epoched: bool,
) -> list[dict[str, Any]]:
    events1 = events_as_list(first.get("event"))
    events2 = events_as_list(second.get("event"))
    merged = [deepcopy(event) for event in events1]
    offset = first_pnts * first_trials
    if not epoched:
        merged.append({"type": "boundary", "latency": first_pnts + 0.5, "duration": np.nan})
    for event in events2:
        updated = deepcopy(event)
        if "latency" in updated:
            updated["latency"] = float(updated["latency"]) + offset
        if "epoch" in updated:
            updated["epoch"] = int(updated["epoch"]) + first_trials
        if "urevent" in updated and not _is_empty(updated["urevent"]):
            updated["urevent"] = (
                int(updated["urevent"]) + len(events_as_list(first.get("urevent"))) + (0 if epoched else 1)
            )
        merged.append(updated)
    return merged


def _merge_urevents(
    first: dict[str, Any],
    second: dict[str, Any],
    first_pnts: int,
    first_trials: int,
    epoched: bool,
) -> list[dict[str, Any]]:
    urevents1 = events_as_list(first.get("urevent"))
    urevents2 = events_as_list(second.get("urevent"))
    merged = [deepcopy(event) for event in urevents1]
    if not epoched:
        merged.append({"type": "boundary", "latency": first_pnts + 0.5})
    offset = first_pnts * first_trials
    for event in urevents2:
        updated = deepcopy(event)
        if "latency" in updated:
            updated["latency"] = float(updated["latency"]) + offset
        merged.append(updated)
    return merged


def _merge_epochs(epochs1: Any, epochs2: Any, first_trials: int, second_trials: int) -> list[dict[str, Any]]:
    first_epochs = _epoch_list(epochs1, first_trials)
    second_epochs = _epoch_list(epochs2, second_trials)
    return first_epochs + second_epochs


def _epoch_list(epochs: Any, trials: int) -> list[dict[str, Any]]:
    if isinstance(epochs, np.ndarray):
        epochs = epochs.tolist()
    if isinstance(epochs, dict):
        epochs = [epochs]
    if isinstance(epochs, list) and epochs:
        return [dict(epoch) for epoch in epochs]
    return [{} for _index in range(trials)]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple, set, dict, str)):
        return len(value) == 0
    return False


def _history_command(indices: list[int] | None, keepall: int | bool) -> str:
    if indices is not None:
        return f"EEG = pop_mergeset( ALLEEG, {format_history_value(indices)}, {int(bool(keepall))});"
    return f"EEG = pop_mergeset( ALLEEG, EEG, {int(bool(keepall))});"


__all__ = ["pop_mergeset", "pop_mergeset_dialog_spec"]
