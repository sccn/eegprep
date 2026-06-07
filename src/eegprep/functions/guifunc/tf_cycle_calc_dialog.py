"""Dialog spec for EEGLAB's time-frequency cycle calculator."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.timefreqfunc.tf_cycle_calc import WIDTH_UNITS

WIDTH_LABELS = (
    "Temporal width FWHM [s]",
    "Spectral bandwidth FWHM [Hz]",
    "Temporal width 2 * sigma_t [s]",
    "Spectral bandwidth 2 * sigma_f [Hz]",
    "Temporal width sigma_t [s]",
    "Spectral bandwidth sigma_f [Hz]",
    "Cycles",
)


def tf_cycle_calc_dialog_spec(
    *,
    freqs: Any = None,
    width: Any = None,
    width_unit: str = "fwhm_t",
) -> DialogSpec:
    """Return the EEGLAB-like ``tf_cycle_calc`` dialog specification."""
    unit_index = WIDTH_UNITS.index(width_unit) + 1 if width_unit in WIDTH_UNITS else 1
    return DialogSpec(
        title="Wavelet cycles calculator -- tf_cycle_calc()",
        controls=(
            ControlSpec("text", "Specify unit:"),
            ControlSpec("popupmenu", "|".join(WIDTH_LABELS), tag="widthpop", value=unit_index),
            ControlSpec("spacer"),
            ControlSpec("text", "Frequency limits [min max] (Hz) or sequence:"),
            ControlSpec("edit", tag="freqedit", value=_format_vector(freqs)),
            ControlSpec("spacer"),
            ControlSpec("text", "(Band-)width limits [min max] (Hz/s/cycles) or sequence:"),
            ControlSpec("edit", tag="widthedit", value=_format_vector(width)),
            ControlSpec("checkbox", "Log spacing", tag="spacingcheck", value=False),
            ControlSpec("spacer"),
            ControlSpec(
                "pushbutton",
                "Plot (band-)width",
                tag="plot",
                callback=CallbackSpec("tf_cycle_calc_plot", params={"width_units": WIDTH_UNITS}),
            ),
            ControlSpec("spacer"),
        ),
        geometry=((3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1)),
        function_name="tf_cycle_calc",
        eeglab_source="functions/timefreqfunc/tf_cycle_calc.m",
        help_text="pophelp('tf_cycle_calc')",
        size=(587, 179),
        row_spacing=4,
    )


def _format_vector(value: Any) -> str:
    if value is None:
        return ""
    values = np.asarray(value, dtype=float).ravel()
    if values.size == 0:
        return ""
    return " ".join(f"{item:g}" for item in values)


__all__ = ["WIDTH_LABELS", "tf_cycle_calc_dialog_spec"]
