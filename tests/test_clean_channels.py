"""clean_channels subset size and the short-FIR noise filter."""

import numpy as np
import pytest
from scipy.signal import fftconvolve

from eegprep.plugins.clean_rawdata.clean_channels import clean_channels
from eegprep.plugins.clean_rawdata.private.sigproc import filtfilt_fast
from eegprep.plugins.firfilt.design import design_fir


def _located_eeg(n_channels: int, n_samples: int, srate: float) -> dict:
    angles = np.linspace(0.0, 2.0 * np.pi, n_channels, endpoint=False)
    data = np.random.default_rng(0).normal(size=(n_channels, n_samples))
    return {
        "data": data,
        "srate": srate,
        "nbchan": n_channels,
        "pnts": n_samples,
        "trials": 1,
        "xmin": 0.0,
        "xmax": (n_samples - 1) / srate,
        "event": [],
        "chanlocs": [
            {
                "labels": f"E{index + 1}",
                "X": float(np.cos(angle)),
                "Y": float(np.sin(angle)),
                "Z": 0.2,
            }
            for index, angle in enumerate(angles)
        ],
    }


def test_subset_size_is_always_a_fraction_of_channel_count():
    """clean_channels.m line 70 always rounds subset_size * nchans."""
    kept = clean_channels(
        _located_eeg(4, 800, 128.0),
        subset_size=0.25,
        corr_threshold=-1.0,
        noise_threshold=100.0,
        num_samples=4,
    )
    assert kept["nbchan"] == 4

    # 2 is not an absolute subset of 2 channels. round(2 * 4) is 8, which
    # cannot be drawn from 4 channels.
    with pytest.raises(IndexError):
        clean_channels(
            _located_eeg(4, 800, 128.0),
            subset_size=2,
            corr_threshold=-1.0,
            noise_threshold=100.0,
            num_samples=4,
        )


def test_short_fir_does_not_spread_a_few_nans_across_the_channel():
    """filter_fast.m uses filter() when length(B) < 256, so NaNs stay local."""
    coefficients = design_fir(100, 2 * np.array([0, 45, 50, 64]) / 128, [1, 1, 0, 0])
    assert coefficients.size == 101
    signal = np.random.default_rng(1).normal(size=2000)
    signal[1000:1003] = np.nan
    filtered = filtfilt_fast(coefficients, 1, signal)
    assert filtered.shape == signal.shape
    assert np.mean(np.isnan(filtered)) < 0.2
    assert np.isfinite(filtered[:200]).all()
    assert np.isfinite(filtered[-200:]).all()


def _legacy_fft_filtfilt(b, x):
    n = len(b)
    padded = np.pad(x, (n, n), mode="reflect", reflect_type="odd")
    forward = fftconvolve(padded, b, mode="full")[::-1]
    filtered = fftconvolve(forward, b, mode="full")[::-1]
    excess = len(filtered) - len(x)
    return filtered[excess // 2 : -excess // 2]


def test_long_fir_keeps_the_fft_path():
    coefficients = np.random.default_rng(2).normal(size=400)
    coefficients /= np.sum(np.abs(coefficients))
    signal = np.random.default_rng(3).normal(size=20000)
    filtered = filtfilt_fast(coefficients, 1, signal)
    np.testing.assert_allclose(filtered, _legacy_fft_filtfilt(coefficients, signal))
