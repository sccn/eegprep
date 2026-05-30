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
    study["etc"] = {**(STUDY.get("etc") or {}), "last_chanplot": last_selection}
    command = _history_command(channels=channels, components=components, measure=measure, mode=mode)
    return (study, command, fig) if return_com else study


def pop_chanplot_dialog_spec(STUDY: dict[str, Any], ALLEEG: Any) -> DialogSpec:
    """Return the STUDY measure plotting dialog spec."""
    datasets = as_eeg_list(ALLEEG)
    labels = channel_labels(datasets[0]) if datasets else []
    design_names = _design_names(STUDY)
    channel_options = ["All channels", *labels] if labels else ["All channels"]
    subject_options = _subject_options(STUDY)
    return DialogSpec(
        title="View and edit current channels -- pop_chanplot()",
        controls=(
            ControlSpec("text", "Select design:", font_weight="bold"),
            ControlSpec("popupmenu", "|".join(design_names), tag="design", value=int(STUDY.get("currentdesign") or 1)),
            ControlSpec("spacer"),
            ControlSpec("text", "Select channel to plot", font_weight="bold"),
            ControlSpec("pushbutton", "Sel. all", tag="select_all_channels", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", "Select subject(s) to plot", font_weight="bold"),
            ControlSpec("listbox", "|".join(channel_options), tag="chan_list", value=[1]),
            ControlSpec(
                "pushbutton",
                "STATS\nparams",
                tag="stats",
                callback=CallbackSpec("set_value", {"source": "stats", "target": "measure_action", "value": "erp"}),
            ),
            ControlSpec("listbox", "|".join(subject_options), tag="chan_onechan", value=[1]),
            _plot_button("Plot ERPs", "erp", "plot_chan_erp"),
            _params_button("erp_params"),
            _plot_button("Plot ERP(s)", "erp", "plot_onechan_erp"),
            _plot_button("Plot spectra", "spec", "plot_chan_spectra"),
            _params_button("spec_params"),
            _plot_button("Plot spectra", "spec", "plot_onechan_spectra"),
            _plot_button("Plot ERPimage", "erpimage", "plot_chan_erpimage", enabled=False),
            _params_button("erpimage_params", enabled=False),
            _plot_button("Plot ERPimage(s)", "erpimage", "plot_onechan_erpimage", enabled=False),
            _plot_button("Plot ERSPs", "ersp", "plot_chan_ersp"),
            _params_button("ersp_params"),
            _plot_button("Plot ERSP(s)", "ersp", "plot_onechan_ersp"),
            _plot_button("Plot ITCs", "itc", "plot_chan_itc"),
            ControlSpec("spacer", tag="measure_action"),
            _plot_button("Plot ITC(s)", "itc", "plot_onechan_itc"),
        ),
        geometry=(
            (0.8, 3),
            (1,),
            (0.6, 0.35, 0.1, 0.1, 0.9),
            (0.9, 0.35, 0.9),
            (0.9, 0.35, 0.9),
            (0.9, 0.35, 0.9),
            (0.9, 0.35, 0.9),
            (0.9, 0.35, 0.9),
            (0.9, 0.35, 0.9),
        ),
        function_name="pop_chanplot",
        eeglab_source="functions/studyfunc/pop_chanplot.m",
        help_text="pophelp('pop_chanplot')",
        size=(770, 500),
        content_margins=(48, 24, 48, 14),
        row_spacing=6,
        geomvert=(1, 0.5, 1, 5, 1, 1, 1, 1, 1),
        button_size=(58, 20),
        extra_stylesheet="""
            QDialog#pop_chanplot QLabel,
            QDialog#pop_chanplot QPushButton,
            QDialog#pop_chanplot QComboBox,
            QDialog#pop_chanplot QListWidget {
                font-size: 12px;
            }
            QDialog#pop_chanplot QListWidget {
                min-height: 126px;
                max-height: 168px;
            }
            QDialog#pop_chanplot QPushButton {
                min-width: 170px;
                max-width: 360px;
                min-height: 20px;
                max-height: 20px;
                padding: 0 8px;
            }
            QDialog#pop_chanplot QPushButton#stats {
                min-width: 82px;
                max-width: 82px;
                min-height: 94px;
                max-height: 126px;
            }
            QDialog#pop_chanplot QPushButton#erp_params,
            QDialog#pop_chanplot QPushButton#spec_params,
            QDialog#pop_chanplot QPushButton#erpimage_params,
            QDialog#pop_chanplot QPushButton#ersp_params {
                min-width: 82px;
                max-width: 82px;
            }
        """,
        known_differences=(
            "EEGPrep keeps the EEGLAB channel-measure action layout and routes OK to the selected/default measure.",
        ),
    )


def _run_gui(STUDY: dict[str, Any], ALLEEG: Any, *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_chanplot_dialog_spec(STUDY, ALLEEG), renderer=renderer)
    if result is None:
        return None
    if "measure" not in result:
        action = str(result.get("measure_action") or "erp")
        selected = numeric_vector(result.get("chan_list", []), dtype=int).tolist()
        channels = [] if not selected or 1 in selected else [index - 1 for index in selected if index > 1]
        return {"channels": channels, "components": [], "measure": _measure_name(action), "mode": "channels"}
    measure_index = _popup_index(result.get("measure"), 1)
    mode_index = _popup_index(result.get("mode"), 1, maximum=2)
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


def _plot_button(label: str, measure: str, tag: str, *, enabled: bool = True) -> ControlSpec:
    return ControlSpec(
        "pushbutton",
        label,
        tag=tag,
        enabled=enabled,
        callback=CallbackSpec("set_value", {"source": tag, "target": "measure_action", "value": measure}),
    )


def _params_button(tag: str, *, enabled: bool = True) -> ControlSpec:
    return ControlSpec("pushbutton", "Params", tag=tag, enabled=enabled)


def _design_names(study: dict[str, Any]) -> list[str]:
    designs = study.get("design") or []
    if isinstance(designs, dict):
        designs = [designs]
    return [str(design.get("name") or f"STUDY.design {index}") for index, design in enumerate(designs, start=1)] or [
        "STUDY.design 1"
    ]


def _subject_options(study: dict[str, Any]) -> list[str]:
    subjects = [str(value) for value in study.get("subject", []) if str(value)]
    return ["All subjects", *subjects] if subjects else ["All subjects"]


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
    if _all_cached_channels_requested(channels):
        return groups
    if isinstance(channels, str):
        try:
            vector = numeric_vector(channels, dtype=int)
        except ValueError:
            return _cached_channel_groups_by_name(groups, [channels])
    elif isinstance(channels, (list, tuple)) and all(isinstance(item, str) for item in channels):
        return _cached_channel_groups_by_name(groups, channels)
    else:
        vector = numeric_vector(channels, dtype=int)
    if vector.size == 0:
        return groups
    if np.any(vector < 1) or np.any(vector > len(groups)):
        raise ValueError(f"channels must be 1-based and within 1..{len(groups)}")
    return [groups[int(index) - 1] for index in vector]


def _cached_channel_groups_by_name(groups: list[dict[str, Any]], channels: list[str]) -> list[dict[str, Any]]:
    lookup = {str(group.get("name") or "").lower(): group for group in groups}
    missing = [label for label in channels if label.lower() not in lookup]
    if missing:
        raise ValueError(f"Unknown precomputed channel group(s): {', '.join(missing)}")
    return [lookup[label.lower()] for label in channels]


def _all_cached_channels_requested(channels: Any) -> bool:
    return channels is None or (isinstance(channels, str) and channels in {"", "channels"})


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
    if isinstance(values, str) and values.lower() == "all":
        return list(range(1, count + 1))
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


def _popup_index(value: Any, default: int, *, maximum: int = len(MEASURE_OPTIONS)) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError):
        index = default
    return max(1, min(index, maximum))


def _history_command(*, channels: Any, components: Any, measure: str, mode: str) -> str:
    pieces = ["STUDY", "ALLEEG"]
    kwargs = {
        "channels": channels,
        "components": components,
        "measure": measure,
        "mode": mode,
    }
    rendered = [python_literal(piece) if piece not in {"STUDY", "ALLEEG"} else piece for piece in pieces]
    rendered.extend(f"{key}={python_literal(value)}" for key, value in kwargs.items() if not _is_empty_arg(value))
    return f"STUDY = pop_chanplot({', '.join(rendered)})"


def _is_empty_arg(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


__all__ = ["pop_chanplot", "pop_chanplot_dialog_spec"]
