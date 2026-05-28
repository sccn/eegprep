"""Minimal STUDY channel-measure plotting handoff."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._plot_utils import (
    as_eeg_list,
    channel_labels,
    eeg_epoch_data,
    eeg_times_ms,
    numeric_vector,
    python_literal,
)


def pop_chanplot(
    STUDY: dict[str, Any] | None = None,
    ALLEEG: Any = None,
    *,
    channels: Any = None,
    measure: str = "erp",
    return_com: bool = False,
):
    """Plot STUDY channel measures from loaded datasets.

    This Phase 4 handoff supports ERP-style channel measure plotting from
    loaded epoched ``ALLEEG`` datasets. Full STUDY precompute/clustering UI
    remains Phase 5 work.
    """
    if STUDY is None:
        raise ValueError("pop_chanplot requires a STUDY structure")
    datasets = as_eeg_list(ALLEEG)
    if not datasets:
        raise ValueError("pop_chanplot requires ALLEEG datasets")
    if measure.lower() != "erp":
        raise NotImplementedError("pop_chanplot currently supports ERP channel measures; STUDY measure UI is Phase 5")
    selected = _selected_channels(channels, int(datasets[0].get("nbchan", 0) or 0))
    times = eeg_times_ms(datasets[0])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = channel_labels(datasets[0])
    for channel in selected:
        erps = []
        for eeg in datasets:
            if int(eeg.get("trials", 1) or 1) <= 1:
                raise ValueError("pop_chanplot ERP measure plotting requires epoched datasets")
            erps.append(np.nanmean(eeg_epoch_data(eeg)[channel, :, :], axis=1))
        ax.plot(times, np.nanmean(np.stack(erps, axis=0), axis=0), label=labels[channel])
    ax.axhline(0, color="0.7", linewidth=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("uV")
    ax.set_title(str(STUDY.get("name") or "STUDY channel ERP"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    STUDY = dict(STUDY)
    STUDY.setdefault("etc", {})["last_chanplot"] = {"measure": measure.lower(), "channels": (selected + 1).tolist()}
    command = f"pop_chanplot(STUDY, ALLEEG, channels={python_literal(channels)}, measure={python_literal(measure)})"
    return (STUDY, command, fig) if return_com else STUDY


def _selected_channels(values: Any, count: int) -> np.ndarray:
    vector = numeric_vector(values, dtype=int)
    if vector.size == 0:
        return np.arange(count)
    if np.any(vector < 1) or np.any(vector > count):
        raise ValueError(f"channels must be 1-based and within 1..{count}")
    return vector - 1


__all__ = ["pop_chanplot"]
