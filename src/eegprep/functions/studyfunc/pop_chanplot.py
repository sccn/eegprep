"""Plot STUDY channel and component measures."""

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
from eegprep.functions.studyfunc.std_readdata import std_readdata


MEASURE_OPTIONS = ("erp", "spec", "ersp", "itc")


def pop_chanplot(
    STUDY: dict[str, Any] | None = None,
    ALLEEG: Any = None,
    *,
    channels: Any = None,
    components: Any = None,
    measure: str = "erp",
    mode: str = "channels",
    gui: bool = False,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Plot precomputed STUDY channel or component measures."""
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
        components = result["components"]
        measure = result["measure"]
        mode = result["mode"]
    measure = _measure_name(measure)
    mode = _mode_name(mode)
    if mode == "channels":
        fig, selected = _plot_channel_measure(STUDY, datasets, channels, measure)
        last_selection = {"measure": measure, "mode": mode, "channels": selected}
    else:
        fig, selected = _plot_component_measure(STUDY, components, measure)
        last_selection = {"measure": measure, "mode": mode, "components": selected}
    study = dict(STUDY)
    study.setdefault("etc", {})["last_chanplot"] = last_selection
    command = _history_command(channels=channels, components=components, measure=measure, mode=mode)
    return (study, command, fig) if return_com else study


def pop_chanplot_dialog_spec(STUDY: dict[str, Any], ALLEEG: Any) -> DialogSpec:
    """Return the STUDY measure plotting dialog spec."""
    datasets = as_eeg_list(ALLEEG)
    labels = channel_labels(datasets[0]) if datasets else []
    return DialogSpec(
        title="Plot channel measures - pop_chanplot()",
        controls=(
            ControlSpec("text", "Measure target"),
            ControlSpec("popupmenu", "Channels|Components", tag="mode", value=1),
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
            ControlSpec("text", "Components ([] = all)"),
            ControlSpec("edit", tag="components", value=""),
            ControlSpec("text", "Measure"),
            ControlSpec("popupmenu", "ERP|Spectrum|ERSP|ITC", tag="measure", value=1),
            ControlSpec("text", f"STUDY: {STUDY.get('name') or ''}"),
        ),
        geometry=((1, 1), (1, 1, 0.22), (1, 1), (1, 1), (1,)),
        function_name="pop_chanplot",
        eeglab_source="functions/studyfunc/pop_chanplot.m",
        help_text="pophelp('pop_chanplot')",
        size=(580, 265),
        known_differences=(
            "EEGPrep uses one dialog to plot cached channel measures and parent-cluster component measures.",
        ),
    )


def _run_gui(STUDY: dict[str, Any], ALLEEG: Any, *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_chanplot_dialog_spec(STUDY, ALLEEG), renderer=renderer)
    if result is None:
        return None
    measure_index = _popup_index(result.get("measure"), 1)
    mode_index = _popup_index(result.get("mode"), 1)
    return {
        "channels": numeric_vector(result.get("channels", []), dtype=int).tolist(),
        "components": numeric_vector(result.get("components", []), dtype=int).tolist(),
        "measure": MEASURE_OPTIONS[measure_index - 1],
        "mode": "components" if mode_index == 2 else "channels",
    }


def _plot_channel_measure(
    study: dict[str, Any], datasets: list[dict[str, Any]], channels: Any, measure: str
) -> tuple[Any, list[int]]:
    if _has_cached_channels(study, measure):
        groups = _cached_channel_groups(study, channels)
        if measure in {"erp", "spec"}:
            fig = _plot_cached_lines(groups, measure, title=str(study.get("name") or "STUDY channel measures"))
        else:
            fig = _plot_cached_image(groups, measure, title=str(study.get("name") or "STUDY channel measures"))
        return fig, [_group_channel_index(group, index) for index, group in enumerate(groups, start=1)]
    if measure != "erp":
        raise ValueError(f"{measure.upper()} channel measures have not been precomputed")
    fig, selected = _plot_loaded_erp(study, datasets, channels)
    return fig, selected


def _plot_component_measure(study: dict[str, Any], components: Any, measure: str) -> tuple[Any, list[int]]:
    _study, data, x_axis, y_axis = std_readdata(study, datatype=measure, clusters=1)
    values = data[0]
    selected = _component_selection(components, values.shape[1] if values.ndim >= 2 else 0)
    indices = np.asarray(selected, dtype=int) - 1
    values = values[:, indices, ...]
    if measure in {"erp", "spec"}:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        y_values = np.nanmean(values, axis=0)
        for index, component_values in zip(selected, y_values):
            ax.plot(x_axis, component_values, label=f"IC {index}")
        ax.set_xlabel("Time (ms)" if measure == "erp" else "Frequency (Hz)")
        ax.set_ylabel("uV" if measure == "erp" else "Power 10*log10(uV^2/Hz)")
        ax.set_title(str(study.get("name") or f"STUDY component {measure.upper()}"))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig, selected
    image = np.nanmean(values, axis=(0, 1))
    return _plot_image(image, x_axis, y_axis, str(study.get("name") or f"STUDY component {measure.upper()}")), selected


def _plot_loaded_erp(study: dict[str, Any], datasets: list[dict[str, Any]], channels: Any) -> tuple[Any, list[int]]:
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
    ax.set_title(str(study.get("name") or "STUDY channel ERP"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, (selected + 1).tolist()


def _plot_cached_lines(groups: list[dict[str, Any]], measure: str, *, title: str) -> Any:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_axis = _axis(groups[0], measure)
    for group in groups:
        data = _data(group, measure)
        ax.plot(x_axis, np.nanmean(data, axis=0), label=str(group.get("name") or "channel"))
    ax.set_xlabel("Time (ms)" if measure == "erp" else "Frequency (Hz)")
    ax.set_ylabel("uV" if measure == "erp" else "Power 10*log10(uV^2/Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_cached_image(groups: list[dict[str, Any]], measure: str, *, title: str) -> Any:
    images = [_data(group, measure) for group in groups]
    image = np.nanmean(np.stack(images, axis=0), axis=(0, 1))
    return _plot_image(image, _axis(groups[0], measure), _freq_axis(groups[0], measure), title)


def _plot_image(image: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, title: str) -> Any:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    mesh = ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=[float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])],
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax)
    fig.tight_layout()
    return fig


def _has_cached_channels(study: dict[str, Any], measure: str) -> bool:
    return any(isinstance(group, dict) and _field(measure) in group for group in study.get("changrp") or [])


def _cached_channel_groups(study: dict[str, Any], channels: Any) -> list[dict[str, Any]]:
    groups = [group for group in study.get("changrp") or [] if isinstance(group, dict)]
    if not groups:
        raise ValueError("No channel measures are stored in STUDY.changrp")
    vector = numeric_vector(channels, dtype=int)
    if vector.size == 0:
        return groups
    if np.any(vector < 1) or np.any(vector > len(groups)):
        raise ValueError(f"channels must be 1-based and within 1..{len(groups)}")
    return [groups[int(index) - 1] for index in vector]


def _data(group: dict[str, Any], measure: str) -> np.ndarray:
    if _field(measure) not in group:
        raise ValueError(f"{measure.upper()} channel measures have not been precomputed")
    return np.asarray(group[_field(measure)], dtype=float)


def _axis(group: dict[str, Any], measure: str) -> np.ndarray:
    field = {"erp": "erptimes", "spec": "specfreqs", "ersp": "ersptimes", "itc": "itctimes"}[measure]
    return np.asarray(group[field], dtype=float)


def _freq_axis(group: dict[str, Any], measure: str) -> np.ndarray:
    field = {"ersp": "erspfreqs", "itc": "itcfreqs"}[measure]
    return np.asarray(group[field], dtype=float)


def _field(measure: str) -> str:
    return {"erp": "erpdata", "spec": "specdata", "ersp": "erspdata", "itc": "itcdata"}[measure]


def _group_channel_index(group: dict[str, Any], fallback: int) -> int:
    inds = group.get("inds") or [fallback]
    return int(inds[0])


def _selected_channels(values: Any, count: int) -> np.ndarray:
    vector = numeric_vector(values, dtype=int)
    if vector.size == 0:
        return np.arange(count)
    if np.any(vector < 1) or np.any(vector > count):
        raise ValueError(f"channels must be 1-based and within 1..{count}")
    return vector - 1


def _component_selection(values: Any, count: int) -> list[int]:
    vector = numeric_vector(values, dtype=int)
    if vector.size == 0:
        return list(range(1, count + 1))
    if np.any(vector < 1) or np.any(vector > count):
        raise ValueError(f"components must be 1-based and within 1..{count}")
    return vector.tolist()


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


def _measure_name(value: Any) -> str:
    text = str(value or "erp").lower()
    aliases = {"spectrum": "spec", "spectra": "spec"}
    text = aliases.get(text, text)
    if text not in MEASURE_OPTIONS:
        raise ValueError("measure must be 'erp', 'spec', 'ersp', or 'itc'")
    return text


def _mode_name(value: Any) -> str:
    text = str(value or "channels").lower()
    if text not in {"channels", "components"}:
        raise ValueError("mode must be 'channels' or 'components'")
    return text


def _popup_index(value: Any, default: int) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError):
        index = default
    return max(1, min(index, len(MEASURE_OPTIONS)))


def _history_command(*, channels: Any, components: Any, measure: str, mode: str) -> str:
    pieces = ["STUDY", "ALLEEG"]
    kwargs = {
        "channels": channels,
        "components": components,
        "measure": measure,
        "mode": mode,
    }
    rendered = [python_literal(piece) if piece not in {"STUDY", "ALLEEG"} else piece for piece in pieces]
    rendered.extend(f"{key}={python_literal(value)}" for key, value in kwargs.items() if value not in (None, []))
    return f"pop_chanplot({', '.join(rendered)})"


__all__ = ["pop_chanplot", "pop_chanplot_dialog_spec"]
