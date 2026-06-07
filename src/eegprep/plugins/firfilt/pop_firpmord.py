"""Standalone pop_firpmord order/weight estimation."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.firfilt._pop_common import numeric_or_none, value_history_command, vector_or_none


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
    """Estimate Parks-McClellan FIR order and pass/stop weights."""
    if gui is None:
        gui = dev is None
    if gui:
        result = inputgui(pop_firpmord_dialog_spec(), renderer=renderer)
        if result is None:
            empty = (None, None, None)
            return (empty, "") if return_com else empty
        f = result.get("f")
        a = result.get("a")
        rp = numeric_or_none(result.get("rp"))
        rs = numeric_or_none(result.get("rs"))
        if rp is None or rs is None:
            raise ValueError("Not enough input arguments.")
        amps = np.asarray(vector_or_none(a), dtype=float)
        dev_values = np.zeros(amps.size, dtype=float)
        dev_values[amps == 1] = (10 ** (rp / 20) - 1) / (10 ** (rp / 20) + 1)
        dev_values[amps == 0] = 10 ** (-abs(rs) / 20)
        dev = dev_values
    fs = 2 if fs is None or str(fs).strip() == "" else float(fs)
    freqs = np.asarray(vector_or_none(f), dtype=float)
    amps = np.asarray(vector_or_none(a), dtype=float)
    devs = np.asarray(vector_or_none(dev), dtype=float)
    if freqs.size == 0 or amps.size == 0 or devs.size == 0:
        raise ValueError("Not enough input arguments")
    if amps.size != devs.size:
        raise ValueError("Desired amplitudes and deviations must have the same length.")
    if freqs.size < 4 or freqs.size % 2:
        raise ValueError("Frequency band edges must contain paired stop/pass band edges.")
    transitions = np.diff(freqs)[1::2]
    min_transition = float(np.min(transitions[transitions > 0]))
    attenuation = -20.0 * np.log10(float(np.min(devs)))
    order = int(np.ceil((attenuation - 8.0) / (2.285 * 2 * np.pi * (min_transition / fs))))
    order = max(2, int(np.ceil(order / 2) * 2))
    weights = np.max(devs) / devs
    wtpass = float(weights[np.where(amps == 1)[0][0]])
    wtstop = float(weights[np.where(amps == 0)[0][0]])
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
        geometry=((1, 1), (1, 1), (1, 1), (1, 1)),
        geomvert=(1, 1, 1, 1),
        size=(560, 232),
        help_text="pophelp('pop_firpmord')",
        controls=(
            ControlSpec("text", "Frequency band edges:"),
            ControlSpec("edit", tag="f", value=""),
            ControlSpec("text", "Desired amplitudes:"),
            ControlSpec("edit", tag="a", value=""),
            ControlSpec("text", "Peak-to-peak passband ripple (dB):"),
            ControlSpec("edit", tag="rp", value=""),
            ControlSpec("text", "Stopband attenuation (dB):"),
            ControlSpec("edit", tag="rs", value=""),
        ),
    )
