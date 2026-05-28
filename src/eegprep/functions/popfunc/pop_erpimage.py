"""EEGLAB-style ERP image wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import (
    component_activations,
    eeg_epoch_data,
    eeg_times_ms,
    history_command,
    numeric_vector,
)
from eegprep.functions.sigprocfunc.erpimage import erpimage


def pop_erpimage(
    EEG: dict[str, Any] | None = None,
    typeplot: int = 1,
    index: int | None = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot an ERP image for one channel or component."""
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
    values = _erpimage_values(EEG, typeplot, int(index))
    figure, image = erpimage(
        values,
        times=eeg_times_ms(EEG),
        title=str(kwargs.pop("title", _default_title(typeplot, int(index)))),
        sort_values=kwargs.pop("sort_values", None),
    )
    command = history_command("pop_erpimage", typeplot, int(index), **kwargs)
    return ({"figure": figure, "image": image}, command) if return_com else {"figure": figure, "image": image}


def pop_erpimage_dialog_spec(EEG: dict[str, Any], *, typeplot: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_erpimage``."""
    is_channel = bool(int(typeplot))
    label = "Channel" if is_channel else "Component(s)"
    title_label = "Channel" if is_channel else "Component"
    smooth = min(max(int(EEG.get("trials", 1) or 1) - 5, 0), 10)
    controls = [
        ControlSpec("text", label, font_weight="bold"),
        ControlSpec("edit", tag="index", value="1"),
        ControlSpec("pushbutton", "...", enabled=is_channel),
        ControlSpec("text", "Fig. title", font_weight="bold"),
        ControlSpec("edit", tag="title", value=""),
    ]
    if not is_channel:
        controls.extend(
            [
                ControlSpec("spacer"),
                ControlSpec("edit", tag="projchan", value=""),
                ControlSpec("spacer"),
                ControlSpec("spacer"),
                ControlSpec("spacer"),
            ]
        )
        controls[5] = ControlSpec("text", "Project to channel #", font_weight="bold")
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
        size=(1113, 834 if is_channel else 870),
    )


def _run_gui(EEG: dict[str, Any], *, typeplot: int, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_erpimage_dialog_spec(EEG, typeplot=typeplot), renderer=renderer)
    if result is None:
        return None
    values = numeric_vector(result.get("index", 1), dtype=int)
    return {
        "index": int(values[0]) if values.size else 1,
        "options": {
            "title": str(result.get("title", "") or ""),
            "smooth": numeric_vector(result.get("smooth", [])).tolist(),
            "decimate": numeric_vector(result.get("decimate", [])).tolist(),
            "limits": numeric_vector(result.get("limtime", [])).tolist(),
            "plotmap": bool(result.get("plotmap", False)),
            "erp": bool(result.get("erp", True)),
            "cbar": bool(result.get("cbar", True)),
            "caxis": numeric_vector(result.get("caxis", [])).tolist(),
            "sortingeventfield": str(result.get("field", "") or ""),
            "sortingtype": str(result.get("type", "") or ""),
            "sortingwin": numeric_vector(result.get("eventrange", [])).tolist(),
            "renorm": str(result.get("renorm", "") or ""),
            "align": numeric_vector(result.get("align", [])).tolist(),
            "nosort": bool(result.get("nosort", False)),
            "noplot": bool(result.get("noplot", False)),
            "phasesort": numeric_vector(result.get("phase", [])).tolist(),
            "coher": numeric_vector(result.get("coher", [])).tolist(),
            "plotamps": bool(result.get("plotamps", False)),
            "spec": numeric_vector(result.get("spec", [])).tolist(),
            "vert": numeric_vector(result.get("vert", [])).tolist(),
            "others": str(result.get("others", "") or ""),
        },
    }


def _erpimage_values(EEG: dict[str, Any], typeplot: int, index: int) -> np.ndarray:
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
    return acts[index - 1, :, :]


def _default_title(typeplot: int, index: int) -> str:
    return f"{'Channel' if typeplot else 'Component'} {index} ERP image"


__all__ = ["pop_erpimage", "pop_erpimage_dialog_spec"]
