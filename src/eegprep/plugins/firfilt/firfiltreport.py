"""FIRFilt report text generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import parse_key_value_args


def firfiltreport(*args: Any, **kwargs: Any) -> str:
    """Return an EEGLAB-style FIR filter report."""
    options = parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True)
    required = ("func", "type", "dir", "order", "family")
    if any(not _has_value(options.get(key)) for key in required):
        raise ValueError("Not enough input arguments.")
    func = str(options["func"])
    ftype = str(options["type"])
    direction = str(options["dir"])
    family = str(options["family"])
    order = int(options["order"])
    is_twopass = direction.startswith("twopass")
    fcatt = -12 if is_twopass else -6
    if is_twopass:
        order *= 2
    lines = [f"{func}() - {ftype} filtering data: {direction}, order {order}, {family}\n"]
    detail_keys = ("fs", "fc", "df", "pbdev", "sbatt")
    if all(_has_value(options.get(key)) for key in detail_keys):
        fs = float(options["fs"])
        fc = np.asarray(options["fc"], dtype=float).ravel()
        df = float(options["df"])
        pbdev = float(options["pbdev"])
        sbatt = float(options["sbatt"])
        fn = fs / 2.0
        dflim = np.asarray([[max(value - df / 2.0, 0.0), min(value + df / 2.0, fn)] for value in fc]).T
        flat = dflim.ravel(order="F")
        if ftype == "lowpass":
            lines.append(f"  cutoff ({fcatt} dB) {_fmt(fc[0])} Hz\n")
            lines.append(
                f"  transition width {df:.1f} Hz, passband 0-{flat[0]:.1f} Hz, stopband {flat[1]:.1f}-{fn:.0f} Hz\n"
            )
        elif ftype == "highpass":
            lines.append(f"  cutoff ({fcatt} dB) {_fmt(fc[0])} Hz\n")
            lines.append(
                f"  transition width {df:.1f} Hz, stopband 0-{flat[0]:.1f} Hz, passband {flat[1]:.1f}-{fn:.0f} Hz\n"
            )
        elif ftype == "bandpass":
            lines.append(f"  cutoff ({fcatt} dB) {_fmt(fc[0])} Hz and {_fmt(fc[1])} Hz\n")
            lines.append(
                f"  transition width {df:.1f} Hz, stopband 0-{flat[0]:.1f} Hz, passband {flat[1]:.1f}-{flat[2]:.1f} Hz, stopband {flat[3]:.1f}-{fn:.0f} Hz\n"
            )
        elif ftype == "bandstop":
            lines.append(f"  cutoff ({fcatt} dB) {_fmt(fc[0])} Hz and {_fmt(fc[1])} Hz\n")
            lines.append(
                f"  transition width {df:.1f} Hz, passband 0-{flat[0]:.1f} Hz, stopband {flat[1]:.1f}-{flat[2]:.1f} Hz, passband {flat[3]:.1f}-{fn:.0f} Hz\n"
            )
        if is_twopass:
            pbdev = (pbdev + 1) ** 2 - 1
            sbatt = sbatt**2
        lines.append(
            f"  max. passband deviation {pbdev:.4f} ({pbdev * 100:.2f}%), stopband attenuation {20 * np.log10(sbatt):.0f} dB\n"
        )
    return "".join(lines)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"
