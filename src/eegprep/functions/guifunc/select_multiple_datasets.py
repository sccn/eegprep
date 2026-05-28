"""EEGLAB-style multi-dataset selection helper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.session import EEGPrepSession, normalize_dataset_indices
from eegprep.functions.popfunc.pop_chansel import pop_chansel


def select_multiple_datasets(
    session: EEGPrepSession,
    indices: Any = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
) -> tuple[list[dict[str, Any]], str] | list[dict[str, Any]]:
    """Select multiple datasets in ``session`` using EEGLAB-facing indices."""
    if gui is None:
        gui = indices is None
    if gui:
        result = _run_gui(session, renderer=renderer)
        if result is None:
            command = ""
            return (
                (session.EEG if isinstance(session.EEG, list) else [session.EEG], command)
                if return_com
                else session.EEG
            )
        indices = result
    selection = normalize_dataset_indices(indices, allow_empty=False)
    eeg = session.retrieve(selection)
    command = _history_command(selection)
    return (eeg if isinstance(eeg, list) else [eeg], command) if return_com else eeg


def _run_gui(session: EEGPrepSession, *, renderer: Any | None = None) -> list[int] | None:
    summaries = session.dataset_summaries()
    if not summaries:
        return None
    dataset_indices = [index for index, _label, _selected in summaries]
    selected = session.selected_dataset_indices() or [dataset_indices[0]]
    selected_positions = [dataset_indices.index(index) + 1 for index in selected if index in dataset_indices]
    labels = [label for _index, label, _selected in summaries]
    chooser = renderer if renderer is not None else pop_chansel
    positions, _selection_text, _all_labels = chooser(
        labels,
        withindex=dataset_indices,
        select=selected_positions,
    )
    if not positions:
        return None
    return [dataset_indices[int(position) - 1] for position in positions]


def _history_command(indices: list[int]) -> str:
    vector = "[" + " ".join(str(index) for index in indices) + "]"
    return f"[ALLEEG EEG CURRENTSET LASTCOM] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', {vector});"


__all__ = ["select_multiple_datasets"]
