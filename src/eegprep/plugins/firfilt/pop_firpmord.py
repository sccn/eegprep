"""Standalone pop_firpmord order/weight estimation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.firfilt._pop_common import numeric_or_none, value_history_command, vector_or_none

# Herrmann, Rabiner, and Chan (1973) coefficients used by MATLAB firpmord/remlpord.
_A1 = 5.309e-3
_A2 = 7.114e-2
_A3 = -4.761e-1
_A4 = -2.66e-3
_A5 = -5.941e-1
_A6 = -4.278e-1
_FK_OFFSET = 11.01217
_FK_SLOPE = 0.51244


def pop_firpmord(
    f: Any = None,
    a: Any = None,
    dev: Any = None,
    fs: Any = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Estimate Parks-McClellan FIR order and pass/stop weights.

    The returned order is ``ceil(L) - 1`` from MATLAB ``firpmord``. Even
    rounding belongs to the ``pop_firpm`` Estimate button, not to this function.
    """
    freqs_input = vector_or_none(f)
    amps_input = vector_or_none(a)
    if freqs_input is None or amps_input is None:
        raise ValueError(
            "pop_firpmord: frequency edges and desired amplitudes are required. "
            "Call pop_firpmord(f, a, dev) with one transition pair per band."
        )
    if gui is None:
        gui = dev is None
    if gui:
        result = inputgui(pop_firpmord_dialog_spec(), renderer=renderer)
        if result is None:
            empty = (None, None, None)
            return (empty, "") if return_com else empty
        rp = numeric_or_none(result.get("rp"))
        rs = numeric_or_none(result.get("rs"))
        if rp is None or rs is None:
            raise ValueError(
                "pop_firpmord: peak-to-peak passband ripple (dB) and stopband attenuation (dB) are required. "
                "Enter both values in the estimate dialog."
            )
        amps = np.asarray(amps_input, dtype=float)
        dev_values = np.zeros(amps.size, dtype=float)
        dev_values[amps == 1] = (10 ** (rp / 20) - 1) / (10 ** (rp / 20) + 1)
        dev_values[amps == 0] = 10 ** (-abs(rs) / 20)
        dev = dev_values
    fs = 2 if fs is None or str(fs).strip() == "" else float(fs)
    freqs = np.asarray(freqs_input, dtype=float)
    amps = np.asarray(amps_input, dtype=float)
    devs = np.asarray(vector_or_none(dev), dtype=float)
    if freqs.size == 0 or amps.size == 0 or devs.size == 0:
        raise ValueError(
            "pop_firpmord: frequency edges, amplitudes, and deviations are required. "
            "Pass dev, or open the dialog and enter ripple and attenuation in dB."
        )
    if amps.size != devs.size:
        raise ValueError(
            "pop_firpmord: amplitudes and deviations must have one value per band. "
            f"Got {amps.size} amplitudes and {devs.size} deviations."
        )
    if freqs.size != 2 * (amps.size - 1):
        raise ValueError(
            "pop_firpmord: frequency edges must be one transition pair per band boundary "
            f"({2 * (amps.size - 1)} edges for {amps.size} bands). "
            "Omit DC and Nyquist; pop_firpm strips them before calling this function."
        )
    if fs <= 0:
        raise ValueError("pop_firpmord: sampling frequency must be positive. Pass fs in Hz, or omit it to use 2.")
    if np.any(devs <= 0):
        raise ValueError(
            "pop_firpmord: each deviation must be positive. "
            "Passband ripple is (10^(rp/20)-1)/(10^(rp/20)+1) and stopband deviation is 10^(-|rs|/20)."
        )
    normalized = freqs / fs
    if np.any(np.diff(normalized) <= 0):
        raise ValueError(
            "pop_firpmord: frequency edges must increase. "
            "Enter each transition as [low high] in Hz, or in fractions of fs."
        )
    lengths = [
        _remlpord_length(
            float(devs[band]), float(devs[band + 1]), float(normalized[2 * band + 1] - normalized[2 * band])
        )
        for band in range(amps.size - 1)
    ]
    order = int(math.ceil(max(lengths)) - 1)
    weights = np.max(devs) / devs
    pass_bands = np.flatnonzero(amps == 1)
    stop_bands = np.flatnonzero(amps == 0)
    if pass_bands.size == 0 or stop_bands.size == 0:
        raise ValueError(
            "pop_firpmord: amplitudes must include a pass band (1) and a stop band (0). For a lowpass, pass [1, 0]."
        )
    wtpass = float(weights[pass_bands[0]])
    wtstop = float(weights[stop_bands[0]])
    output = (order, wtpass, wtstop)
    command = value_history_command(
        "pop_firpmord", [freqs.tolist(), amps.tolist(), devs.tolist(), fs], assignment="[m, wtpass, wtstop]"
    )
    return (output, command) if return_com else output


def pop_firpmord_dialog_spec() -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_firpmord``."""
    return DialogSpec(
        title="Estimate filter order and weights -- pop_firpmord()",
        function_name="pop_firpmord",
        eeglab_source="plugins/firfilt/pop_firpmord.m",
        geometry=((1, 1), (1, 1)),
        geomvert=(1, 1),
        size=(560, 176),
        help_text="pophelp('pop_firpmord')",
        controls=(
            ControlSpec("text", "Peak-to-peak passband ripple (dB):"),
            ControlSpec("edit", tag="rp", value=""),
            ControlSpec("text", "Stopband attenuation (dB):"),
            ControlSpec("edit", tag="rs", value=""),
        ),
    )


def _remlpord_length(d1: float, d2: float, df: float) -> float:
    """Return the Herrmann filter length L for one transition of width ``df``."""
    log_d1 = math.log10(d1)
    log_d2 = math.log10(d2)
    dinf = (_A1 * log_d1**2 + _A2 * log_d1 + _A3) * log_d2 + (_A4 * log_d1**2 + _A5 * log_d1 + _A6)
    fk = _FK_OFFSET + _FK_SLOPE * (log_d1 - log_d2)
    return dinf / df - fk * df + 1.0
