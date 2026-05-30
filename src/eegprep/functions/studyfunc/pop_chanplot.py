"""Minimal STUDY channel-measure plotting handoff."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
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
    gui: bool = False,
    renderer: Any | None = None,
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
    if gui:
        result = _run_gui(STUDY, datasets, renderer=renderer)
        if result is None:
            return (STUDY, "", None) if return_com else STUDY
        channels = result["channels"]
        measure = result["measure"]
    if measure.lower() != "erp":
        raise NotImplementedError("pop_chanplot currently supports ERP channel measures; STUDY measure UI is Phase 5")
    _validate_time_grid(datasets)
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


def pop_chanplot_dialog_spec(STUDY: dict[str, Any], ALLEEG: Any) -> DialogSpec:
    """Return the Phase 4 STUDY channel-measure dialog spec."""
    datasets = as_eeg_list(ALLEEG)
    labels = channel_labels(datasets[0]) if datasets else []
    return DialogSpec(
        title="Plot channel measures - pop_chanplot()",
        controls=(
            ControlSpec("text", "Channels ([] = all)"),
            ControlSpec("edit", tag="channels", value=""),
            ControlSpec(
                "pushbutton",
                "...",
                tag="channels_button",
                callback=CallbackSpec(
                    "select_channels",
                    params={
                        "button": "channels_button",
                        "target": "channels",
                        "channels": labels,
                        "return_indices": True,
                    },
                    matlab_callback="pop_chansel({tmpchanlocs.labels}, 'withindex', 'on')",
                ),
            ),
            ControlSpec("text", "Measure"),
            ControlSpec("popupmenu", "ERP", tag="measure", value=1),
            ControlSpec("text", f"STUDY: {STUDY.get('name') or ''}"),
        ),
        geometry=((1, 1, 0.22), (1, 1), (1,)),
        function_name="pop_chanplot",
        eeglab_source="functions/studyfunc/pop_chanplot.m",
        help_text="pophelp('pop_chanplot')",
        size=(520, 215),
    )


def _run_gui(STUDY: dict[str, Any], ALLEEG: Any, *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_chanplot_dialog_spec(STUDY, ALLEEG), renderer=renderer)
    if result is None:
        return None
    measure_options = ["erp"]
    try:
        measure_index = int(result.get("measure", 1)) - 1
    except (TypeError, ValueError):
        measure_index = 0
    measure = measure_options[measure_index] if 0 <= measure_index < len(measure_options) else "erp"
    return {"channels": numeric_vector(result.get("channels", []), dtype=int).tolist(), "measure": measure}


def _selected_channels(values: Any, count: int) -> np.ndarray:
    vector = numeric_vector(values, dtype=int)
    if vector.size == 0:
        return np.arange(count)
    if np.any(vector < 1) or np.any(vector > count):
        raise ValueError(f"channels must be 1-based and within 1..{count}")
    return vector - 1


def _validate_time_grid(datasets: list[dict[str, Any]]) -> None:
    reference = (
        int(datasets[0].get("pnts", 0) or 0),
        float(datasets[0].get("srate", 0) or 0),
        float(datasets[0].get("xmin", 0) or 0),
        float(datasets[0].get("xmax", 0) or 0),
    )
    for index, eeg in enumerate(datasets[1:], start=2):
        candidate = (
            int(eeg.get("pnts", 0) or 0),
            float(eeg.get("srate", 0) or 0),
            float(eeg.get("xmin", 0) or 0),
            float(eeg.get("xmax", 0) or 0),
        )
        if candidate != reference:
            raise ValueError(f"Dataset {index} does not share the same time grid")


__all__ = ["pop_chanplot", "pop_chanplot_dialog_spec"]
