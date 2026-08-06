"""EEGLAB-style ERP image wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc.plot_utils import (
    channel_labels,
    component_activations,
    component_maps,
    eeg_epoch_data,
    eeg_times_ms,
    history_command,
    numeric_vector,
    parse_plot_options_text,
    show_figures,
)
from eegprep.functions.popfunc._pop_utils import is_on
from eegprep.functions.sigprocfunc.erpimage import erpimage


def pop_erpimage(
    EEG: dict[str, Any] | None = None,
    typeplot: int = 1,
    index: int | None = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str = "on",
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot an ERP image for one channel or component.

    Pass ``plot='off'`` to build and return the figure without opening a window.
    """
    if EEG is None:
        return (None, "") if return_com else None
    typeplot = int(typeplot)
    if gui is None:
        gui = index is None
    if gui:
        result = _run_gui(EEG, typeplot=typeplot, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        index = result["index"]
        kwargs.update(result["options"])
    if index is None:
        index = 1
    _raise_for_unsupported_kwargs(kwargs)
    command_kwargs = dict(kwargs)
    projchan = kwargs.pop("projchan", None)
    values = _erpimage_values(EEG, typeplot, int(index), projchan=projchan)
    times = eeg_times_ms(EEG)
    sort_values = kwargs.pop("sort_values", None)
    sorting_field = kwargs.pop("sortingeventfield", kwargs.pop("field", ""))
    sorting_type = kwargs.pop("sortingtype", kwargs.pop("type", []))
    sorting_window = kwargs.pop("sortingwin", kwargs.pop("eventrange", []))
    renorm = kwargs.pop("renorm", "no")
    if sorting_field:
        sort_values = _event_sort_values(EEG, sorting_field, sorting_type, sorting_window, renorm)
    if is_on(kwargs.pop("nosort", False)):
        sort_values = None
    kwargs.pop("noplot", None)
    align = numeric_vector(kwargs.pop("align", []))
    if align.size:
        raise ValueError("pop_erpimage event alignment is not available in EEGPrep's standalone ERP image plot")
    limits = numeric_vector(kwargs.pop("limits", []))
    if limits.size == 2:
        mask = (times >= limits[0]) & (times <= limits[1])
        if not np.any(mask):
            raise ValueError("limits do not contain any samples")
        values = values[mask, :]
        times = times[mask]
    figure, image = erpimage(
        values,
        times=times,
        title=str(kwargs.pop("title", _default_title(typeplot, int(index), projchan=projchan))),
        sort_values=sort_values,
        smooth=kwargs.pop("smooth", None),
        decimate=_first_int(kwargs.pop("decimate", None), default=1),
        caxis=kwargs.pop("caxis", None),
        cbar=bool(kwargs.pop("cbar", True)),
        plot_erp=bool(kwargs.pop("erp", True)),
        vert=kwargs.pop("vert", None),
    )
    command = history_command("pop_erpimage", typeplot, int(index), **command_kwargs)
    show_figures(figure, plot=plot)
    return ({"figure": figure, "image": image}, command) if return_com else {"figure": figure, "image": image}


def pop_erpimage_dialog_spec(EEG: dict[str, Any], *, typeplot: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_erpimage``."""
    is_channel = bool(int(typeplot))
    label = "Channel" if is_channel else "Component(s)"
    title_label = "Channel" if is_channel else "Component"
    smooth = min(max(int(EEG.get("trials", 1) or 1) - 5, 0), 10)
    labels = channel_labels(EEG)
    channel_selector = ControlSpec(
        "pushbutton",
        "...",
        tag="index_button",
        enabled=is_channel and bool(labels),
        callback=CallbackSpec(
            "select_channels",
            params={
                "button": "index_button",
                "target": "index",
                "channels": labels,
                "selectionmode": "single",
                "return_indices": True,
            },
            matlab_callback="pop_chansel({tmpchanlocs.labels}, 'withindex', 'on', 'selectionmode', 'single')",
        ),
    )
    controls: list[ControlSpec] = [
        ControlSpec("text", label, font_weight="bold"),
        ControlSpec("edit", tag="index", value="1"),
    ]
    if is_channel:
        controls.append(channel_selector)
    else:
        controls.append(ControlSpec("spacer"))
        controls.extend(
            [
                ControlSpec("spacer"),
                ControlSpec("spacer"),
                ControlSpec("text", "Project to channel #", font_weight="bold"),
                ControlSpec("edit", tag="projchan", value=""),
                ControlSpec(
                    "pushbutton",
                    "...",
                    tag="projchan_button",
                    enabled=bool(labels),
                    callback=CallbackSpec(
                        "select_channels",
                        params={
                            "button": "projchan_button",
                            "target": "projchan",
                            "channels": labels,
                            "selectionmode": "single",
                            "return_indices": True,
                        },
                        matlab_callback=(
                            "pop_chansel({tmpchanlocs.labels}, 'withindex', 'on', 'selectionmode', 'single')"
                        ),
                    ),
                ),
                ControlSpec("text", "Fig. title", font_weight="bold"),
                ControlSpec("edit", tag="title", value=""),
            ]
        )
    if is_channel:
        controls.extend(
            [
                ControlSpec("text", "Fig. title", font_weight="bold"),
                ControlSpec("edit", tag="title", value=""),
            ]
        )
    controls.extend(
        [
            ControlSpec("text", "Smoothing", font_weight="bold"),
            ControlSpec("edit", tag="smooth", value=str(smooth)),
            ControlSpec("checkbox", "Plot scalp map", tag="plotmap", value=True),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", "Downsampling", font_weight="bold"),
            ControlSpec("edit", tag="decimate", value="1"),
            ControlSpec("checkbox", "Plot ERP", tag="erp", value=True),
            ControlSpec("text", "ERP limits (uV)" if is_channel else "ERP limits"),
            ControlSpec("edit", tag="limerp", value=""),
            ControlSpec("text", "Time limits (ms)", font_weight="bold"),
            ControlSpec(
                "edit",
                tag="limtime",
                value=f"{float(EEG.get('xmin', 0)) * 1000:g} {float(EEG.get('xmax', 0)) * 1000:g}",
            ),
            ControlSpec("checkbox", "Plot colorbar", tag="cbar", value=True),
            ControlSpec("text", "Color limits (see Help)"),
            ControlSpec("edit", tag="caxis", value=""),
            ControlSpec("spacer"),
            ControlSpec("text", "Sort/align trials by epoch event values", font_weight="bold"),
            ControlSpec("pushbutton", "Epoch-sorting field"),
            ControlSpec("pushbutton", "Event type(s)"),
            ControlSpec("text", "Event time range"),
            ControlSpec("text", "Rescale"),
            ControlSpec("text", "Align"),
            ControlSpec("checkbox", "Don't sort by value", tag="nosort", value=False),
            ControlSpec("edit", tag="field", value=""),
            ControlSpec("edit", tag="type", value=""),
            ControlSpec("edit", tag="eventrange", value=""),
            ControlSpec("edit", tag="renorm", value="no"),
            ControlSpec("edit", tag="align", value=""),
            ControlSpec("checkbox", "Don't plot values", tag="noplot", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Sort trials by phase", font_weight="bold"),
            ControlSpec("text", "Frequency (Hz | minHz maxHz)"),
            ControlSpec("text", "Percent low-amp. trials to ignore"),
            ControlSpec("text", "Window center (ms)"),
            ControlSpec("text", "Wavelet cycles"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="phase", value=""),
            ControlSpec("edit", tag="phase2", value=""),
            ControlSpec("edit", tag="phase3", value=""),
            ControlSpec("text", "        3"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", "Inter-trial coherence options", font_weight="bold"),
            ControlSpec("text", "Frequency (Hz | minHz maxHz)"),
            ControlSpec("text", "Signif. level (<0.20)"),
            ControlSpec("text", "Amplitude limits (dB)"),
            ControlSpec("text", "Coher limits (<=1)"),
            ControlSpec("checkbox", "Image amps", tag="plotamps", value=False),
            ControlSpec("edit", tag="coher", value=""),
            ControlSpec("edit", tag="coher2", value=""),
            ControlSpec("edit", tag="limamp", value=""),
            ControlSpec("edit", tag="limcoher", value=""),
            ControlSpec("text", "   (Requires signif.)"),
            ControlSpec("spacer"),
            ControlSpec("text", "Other options", font_weight="bold"),
            ControlSpec("text", "Plot spectrum (minHz maxHz)"),
            ControlSpec("text", "Baseline ampl. (dB)"),
            ControlSpec("text", "Mark times (ms)"),
            ControlSpec("text", "More options (see >> help erpimage)"),
            ControlSpec("edit", tag="spec", value=""),
            ControlSpec("edit", tag="limbaseamp", value=""),
            ControlSpec("edit", tag="vert", value=""),
            ControlSpec("edit", tag="others", value=""),
        ]
    )
    geometry = [
        (1, 1, 0.4, 0.5, 2.1),
        (1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1),
        (1,),
        (1,),
        (1, 1, 1, 0.8, 0.8, 1.2),
        (1, 1, 1, 0.8, 0.8, 1.2),
        (1,),
        (1,),
        (1.6, 1.7, 1.2, 1, 0.5),
        (1.6, 1.7, 1.2, 1, 0.5),
        (1,),
        (1,),
        (1.5, 1, 1, 1, 1),
        (1.5, 1, 1, 1, 1),
        (1,),
        (1,),
        (1.5, 1, 1, 2.2),
        (1.5, 1, 1, 2.2),
    ]
    if not is_channel:
        geometry.insert(0, (1, 1, 0.1, 0.8, 2.1))
    return DialogSpec(
        title=f"{title_label} ERP image -- pop_erpimage()",
        controls=tuple(controls),
        geometry=tuple(geometry),
        function_name="pop_erpimage",
        eeglab_source="functions/popfunc/pop_erpimage.m",
        help_text="pophelp('pop_erpimage')",
        size=(1113, 831),
        row_spacing=4,
        extra_stylesheet="""
            QDialog#pop_erpimage QLabel,
            QDialog#pop_erpimage QCheckBox,
            QDialog#pop_erpimage QLineEdit,
            QDialog#pop_erpimage QPushButton {
                font-size: 11px;
            }
            QDialog#pop_erpimage QLineEdit,
            QDialog#pop_erpimage QPushButton {
                min-height: 15px;
                max-height: 15px;
            }
            QDialog#pop_erpimage QCheckBox::indicator {
                width: 11px;
                height: 11px;
            }
        """,
    )


def _run_gui(EEG: dict[str, Any], *, typeplot: int, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_erpimage_dialog_spec(EEG, typeplot=typeplot), renderer=renderer)
    if result is None:
        return None
    _raise_for_unsupported_gui_options(result)
    other_options = _normalise_other_options(result.get("others", ""))
    values = numeric_vector(result.get("index", 1), dtype=int)
    options = {
        "title": str(result.get("title", "") or ""),
        "smooth": numeric_vector(result.get("smooth", [])).tolist(),
        "decimate": numeric_vector(result.get("decimate", [])).tolist(),
        "limits": numeric_vector(result.get("limtime", [])).tolist(),
        "erp": bool(result.get("erp", True)),
        "cbar": bool(result.get("cbar", True)),
        "caxis": numeric_vector(result.get("caxis", [])).tolist(),
        "vert": numeric_vector(result.get("vert", [])).tolist(),
    }
    field = str(result.get("field", "") or "").strip()
    if field:
        options["sortingeventfield"] = field
        event_type = str(result.get("type", "") or "").strip()
        if event_type:
            options["sortingtype"] = _parse_event_type_text(event_type)
        eventrange = numeric_vector(result.get("eventrange", [])).tolist()
        if eventrange:
            options["sortingwin"] = eventrange
        renorm = str(result.get("renorm", "") or "no").strip()
        if renorm and renorm.lower() != "no":
            options["renorm"] = renorm
    if bool(result.get("nosort", False)):
        options["nosort"] = True
    if bool(result.get("noplot", False)):
        options["noplot"] = True
    options.update(other_options)
    projchan = numeric_vector(result.get("projchan", []), dtype=int)
    if projchan.size:
        options["projchan"] = projchan.tolist()
    return {
        "index": int(values[0]) if values.size else 1,
        "options": options,
    }


def _erpimage_values(EEG: dict[str, Any], typeplot: int, index: int, *, projchan: Any = None) -> np.ndarray:
    if int(EEG.get("trials", 1) or 1) <= 1:
        raise ValueError("pop_erpimage requires epoched data")
    if typeplot:
        data = eeg_epoch_data(EEG)
        if index < 1 or index > data.shape[0]:
            raise ValueError("channel index is outside available channels")
        return data[index - 1, :, :]
    acts = component_activations(EEG)
    if index < 1 or index > acts.shape[0]:
        raise ValueError("component index is outside available ICA components")
    values = acts[index - 1, :, :]
    proj_indices = numeric_vector(projchan, dtype=int)
    if proj_indices.size == 0:
        return values
    maps = component_maps(EEG)
    if np.any(proj_indices < 1) or np.any(proj_indices > maps.shape[0]):
        raise ValueError(f"projected channel indices must be 1-based and within 1..{maps.shape[0]}")
    weights = maps[proj_indices - 1, index - 1]
    projected = values[np.newaxis, :, :] * weights[:, np.newaxis, np.newaxis]
    return np.nanmean(projected, axis=0)


def _default_title(typeplot: int, index: int, *, projchan: Any = None) -> str:
    if typeplot:
        return f"Channel {index} ERP image"
    proj_indices = numeric_vector(projchan, dtype=int)
    if proj_indices.size:
        return f"Component {index} -> Channel {' '.join(str(value) for value in proj_indices.tolist())} ERP image"
    return f"Component {index} ERP image"


def _first_int(value: Any, *, default: int) -> int:
    vector = numeric_vector(value, dtype=int)
    if vector.size == 0:
        return default
    return int(vector[0])


def _raise_for_unsupported_gui_options(result: dict[str, Any]) -> None:
    unsupported_text_fields = {
        "align": "event alignment",
        "phase": "phase sorting",
        "phase2": "phase sorting",
        "phase3": "phase sorting",
        "coher": "inter-trial coherence",
        "coher2": "inter-trial coherence",
        "limamp": "inter-trial coherence amplitude limits",
        "limcoher": "inter-trial coherence limits",
        "spec": "spectrum inset",
        "limbaseamp": "baseline amplitude limits",
    }
    for field, label in unsupported_text_fields.items():
        if str(result.get(field, "") or "").strip():
            raise ValueError(f"pop_erpimage {label} is not available in EEGPrep's standalone ERP image plot")
    unsupported_checks = {
        "plotamps": "amplitude image mode",
    }
    for field, label in unsupported_checks.items():
        if bool(result.get(field, False)):
            raise ValueError(f"pop_erpimage {label} is not available in EEGPrep's standalone ERP image plot")


def _raise_for_unsupported_kwargs(kwargs: dict[str, Any]) -> None:
    supported = {
        "title",
        "sort_values",
        "smooth",
        "decimate",
        "limits",
        "caxis",
        "cbar",
        "erp",
        "vert",
        "projchan",
        "sortingeventfield",
        "sortingtype",
        "sortingwin",
        "field",
        "type",
        "eventrange",
        "renorm",
        "nosort",
        "noplot",
        "align",
    }
    unsupported = sorted(set(kwargs) - supported)
    if unsupported:
        raise ValueError(
            "pop_erpimage option(s) are not available in EEGPrep's standalone ERP image plot: " + ", ".join(unsupported)
        )


def _event_sort_values(EEG: dict[str, Any], field: Any, event_types: Any, eventrange: Any, renorm: Any) -> np.ndarray:
    field_name = str(field).strip()
    if not field_name:
        return np.asarray([])
    if str(renorm).strip().lower() not in {"", "no", "yes"}:
        raise ValueError("pop_erpimage renorm formulas are not available in EEGPrep; use 'yes' or 'no'")
    trials = int(EEG.get("trials", 1) or 1)
    pnts = int(EEG.get("pnts", 0) or 0)
    srate = float(EEG.get("srate", 1) or 1)
    xmin = float(EEG.get("xmin", 0) or 0)
    types = _event_type_set(event_types)
    window = numeric_vector(eventrange)
    if window.size not in {0, 2}:
        raise ValueError("sortingwin/eventrange must contain [start end] in milliseconds")
    values = np.full(trials, np.nan)
    needs_latency = window.size == 2 or field_name == "latency"
    for event in _event_records(EEG.get("event")):
        epoch = _event_epoch(event, pnts)
        if epoch < 1 or epoch > trials:
            continue
        if types and str(event.get("type", "")) not in types:
            continue
        latency_ms = _event_latency_ms(event, epoch, pnts, srate, xmin) if needs_latency else np.nan
        if window.size == 2 and not (window[0] <= latency_ms <= window[1]):
            continue
        if not np.isnan(values[epoch - 1]):
            continue
        raw_value = latency_ms if field_name == "latency" else event.get(field_name, np.nan)
        try:
            values[epoch - 1] = float(raw_value)
        except (TypeError, ValueError):
            values[epoch - 1] = np.nan
    if np.all(np.isnan(values)):
        raise ValueError("pop_erpimage event sorting found no matching event values")
    if str(renorm).strip().lower() == "yes":
        finite = np.isfinite(values)
        if np.any(finite):
            minimum = float(np.nanmin(values))
            maximum = float(np.nanmax(values))
            if maximum > minimum:
                values[finite] = (values[finite] - minimum) / (maximum - minimum)
    return values


def _event_records(events: Any) -> list[dict[str, Any]]:
    if events is None:
        return []
    if isinstance(events, np.ndarray):
        events = events.tolist()
    if isinstance(events, dict):
        return [dict(events)]
    return [dict(event) for event in events]


def _event_epoch(event: dict[str, Any], pnts: int) -> int:
    if event.get("epoch") not in (None, ""):
        return int(float(event["epoch"]))
    if pnts <= 0:
        return 1
    return int(np.floor((float(event.get("latency", 1)) - 1) / pnts)) + 1


def _event_latency_ms(event: dict[str, Any], epoch: int, pnts: int, srate: float, xmin: float) -> float:
    latency = float(event.get("latency", 1))
    sample_in_epoch = latency - (epoch - 1) * pnts
    return ((sample_in_epoch - 1) / srate + xmin) * 1000.0


def _event_type_set(value: Any) -> set[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return set()
    if isinstance(value, np.ndarray):
        value = value.ravel().tolist()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def _parse_event_type_text(text: str) -> list[str]:
    return [token.strip("'\"") for token in text.replace(",", " ").split() if token.strip("'\"")]


def _normalise_other_options(text: Any) -> dict[str, Any]:
    parsed = parse_plot_options_text(text)
    options: dict[str, Any] = {}
    aliases = {"avewidth": "smooth", "limit": "limits"}
    for key, value in parsed.items():
        normalised = aliases.get(key, key)
        if normalised in {"erp", "cbar", "nosort", "noplot"}:
            options[normalised] = is_on(value)
        elif normalised in {"title", "smooth", "decimate", "limits", "caxis", "vert"}:
            options[normalised] = value
        else:
            raise ValueError(f"pop_erpimage More options does not support '{key}' in EEGPrep")
    return options


__all__ = ["pop_erpimage", "pop_erpimage_dialog_spec"]
