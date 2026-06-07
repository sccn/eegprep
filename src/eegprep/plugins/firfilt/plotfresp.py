"""Frequency-response plotting for FIRFilt filters."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import dimpulse


def plotfresp(
    b: Any,
    a: Any = 1,
    nfft: int = 512,
    fs: float = 1,
    dir: str = "onepass",
    *,
    show: bool = True,
):
    """Plot impulse, step, magnitude, and phase response like FIRFilt."""
    response = _response_data(b, a, nfft, fs, dir)
    fig = plt.figure()
    fig.set_size_inches(8, 5)
    axes = [
        fig.add_subplot(2, 3, 1),
        fig.add_subplot(2, 3, 2),
        fig.add_subplot(2, 3, 3),
        fig.add_subplot(2, 3, 4),
        fig.add_subplot(2, 3, 5),
    ]
    axes[0].stem(response["impulse_x"], response["impulse_response"])
    axes[0].set_title("Impulse response", fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[3].stem(response["impulse_x"], response["step_response"])
    axes[3].set_title("Step response", fontweight="bold")
    axes[3].set_ylabel("Amplitude")
    axes[1].plot(response["frequency_hz"], response["magnitude_linear"])
    axes[1].set_title("Magnitude response", fontweight="bold")
    axes[1].set_ylabel("Magnitude (linear)")
    axes[4].plot(response["frequency_hz"], response["magnitude_db"])
    axes[4].set_title("Magnitude response", fontweight="bold")
    axes[4].set_ylabel("Magnitude (dB)")
    axes[2].plot(response["frequency_hz"], response["phase_rad"])
    axes[2].set_title("Phase response", fontweight="bold")
    axes[2].set_ylabel("Phase (rad)")
    freq_label = "Normalized frequency (2 pi rad / sample)" if float(fs) == 1 else "Frequency (Hz)"
    for axis in (axes[1], axes[2], axes[4]):
        axis.set_xlabel(freq_label)
        axis.set_xlim(0, float(fs) / 2)
        axis.grid(True)
    for axis in (axes[0], axes[3]):
        axis.set_xlabel("n (samples)")
        axis.grid(True)
    fig.tight_layout()
    if show:
        plt.show(block=False)
    return fig, axes, response


def _response_data(b: Any, a: Any, nfft: int, fs: float, direction: str) -> dict[str, np.ndarray]:
    b = np.asarray(b, dtype=float).ravel()
    a = np.asarray(a, dtype=float).ravel()
    if b.size == 0:
        raise ValueError("Not enough input arguments.")
    nfft = int(nfft or 512)
    fs = float(fs or 1)
    direction = str(direction or "onepass")
    is_fir = a.size == 1 and a[0] == 1
    is_linear_phase = is_fir and np.all(b == b[::-1])
    if direction.startswith("twopass"):
        is_twopass = True
        is_zerophase = True
    elif direction == "onepass-zerophase":
        if not is_linear_phase:
            raise ValueError("Onepass-zerophase filtering is only allowed for linear-phase FIR filters.")
        is_twopass = False
        is_zerophase = True
    else:
        is_twopass = False
        is_zerophase = False
    if is_fir:
        impulse = b.copy()
    else:
        _time, values = dimpulse((b, a, 1), n=max(nfft, b.size))
        impulse = np.asarray(values[0]).ravel()
    if is_twopass:
        impulse = np.convolve(impulse, impulse[::-1])
    n = impulse.size
    x = np.arange(n, dtype=float)
    if is_zerophase:
        group_delay = (n - 1) / 2
        x = np.arange(-group_delay, group_delay + 1)
    nfft = max(2 ** int(np.ceil(np.log2(n))), nfft)
    z = np.fft.fft(impulse, nfft)[: nfft // 2 + 1]
    freq = np.linspace(0, fs / 2, nfft // 2 + 1)
    phase = np.unwrap(np.angle(z))
    if is_zerophase:
        delay = -freq / fs * ((n - 1) / 2) * 2 * np.pi
        phase = phase - delay
        phase = np.mod(np.round(phase / np.pi), 2) * np.pi
    magnitude = np.abs(z)
    return {
        "impulse_x": x,
        "impulse_response": impulse,
        "step_response": np.cumsum(impulse),
        "frequency_hz": freq,
        "magnitude_linear": magnitude,
        "magnitude_db": 20 * np.log10(np.maximum(magnitude, np.finfo(float).tiny)),
        "phase_rad": phase,
    }
