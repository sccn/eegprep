"""clean_channels subset size and the short-FIR noise filter."""

import os

import numpy as np
import pytest
from scipy.signal import fftconvolve

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
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
    with pytest.warns(UserWarning, match="planar"):
        kept = clean_channels(
            _located_eeg(4, 800, 128.0),
            subset_size=0.25,
            corr_threshold=-1.0,
            noise_threshold=100.0,
            num_samples=4,
        )
    assert kept["nbchan"] == 4

    # clean_channels.m:70 gives round(2 * 4) = 8. randsample then indexes past
    # the 4-channel pool, so this is an error rather than a subset of 2.
    with pytest.raises(ValueError, match="more than the 4 channels"):
        clean_channels(
            _located_eeg(4, 800, 128.0),
            subset_size=2,
            corr_threshold=-1.0,
            noise_threshold=100.0,
            num_samples=4,
        )


def _short_fir_nan_mask(length: int, nan_start: int, nan_stop: int, coefficient_count: int) -> np.ndarray:
    """NaNs an FIR filtfilt can produce from one interior run.

    ``lfilter`` of ``coefficient_count`` taps reads that many samples, so one
    pass reaches ``coefficient_count - 1`` samples past a NaN. The return pass
    reaches the same distance the other way. The run is far enough from either
    end that the odd-reflection pad does not carry a NaN back in.
    """
    reach = coefficient_count - 1
    if nan_start < reach or length - nan_stop < reach:
        raise ValueError("NaN run is close enough to an edge for the pad to matter")
    mask = np.zeros(length, dtype=bool)
    mask[nan_start - reach : nan_stop + reach] = True
    return mask


def test_short_fir_does_not_spread_a_few_nans_across_the_channel():
    """filter_fast.m uses filter() when length(B) < 256, so NaNs stay local."""
    coefficients = design_fir(100, 2 * np.array([0, 45, 50, 64]) / 128, [1, 1, 0, 0])
    assert coefficients.size == 101
    signal = np.random.default_rng(1).normal(size=2000)
    signal[1000:1003] = np.nan
    filtered = filtfilt_fast(coefficients, 1, signal)
    assert filtered.shape == signal.shape
    expected = _short_fir_nan_mask(signal.size, 1000, 1003, coefficients.size)
    np.testing.assert_array_equal(np.isnan(filtered), expected)


@pytest.mark.skipif(os.getenv("EEGPREP_SKIP_MATLAB") == "1", reason="MATLAB not available")
def test_parity_short_fir_nans_match_matlab():
    try:
        eeglab = get_eeglab("MAT")
    except Exception as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    coefficients = design_fir(100, 2 * np.array([0, 45, 50, 64]) / 128, [1, 1, 0, 0])
    signal = np.random.default_rng(1).normal(size=2000)
    signal[1000:1003] = np.nan
    filtered = filtfilt_fast(coefficients, 1, signal)
    # filtfilt_fast.m filters along columns. clean_channels.m:92 passes
    # signal.data(c,:)', so a 1-D row would be read as one sample per column.
    matlab = np.asarray(eeglab.filtfilt_fast(coefficients, 1, signal[:, None]), dtype=float).ravel()
    expected = _short_fir_nan_mask(signal.size, 1000, 1003, coefficients.size)
    np.testing.assert_array_equal(np.isnan(matlab), expected)
    np.testing.assert_allclose(filtered, matlab, atol=1e-8, rtol=1e-8, equal_nan=True)


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


@pytest.mark.parametrize("missing_coord", ["", [], (), np.array([]), None])
def test_unlocated_auxiliary_channels_do_not_break_clean_channels(missing_coord):
    """Auxiliary channels (e.g. EOG) with blank or empty X/Y/Z coordinates are excluded from usable_channels."""
    eeg = _located_eeg(8, 800, 128.0)
    # Give the 6 located channels 3-D extent so validate_electrode_positions does not warn
    for idx in range(6):
        eeg["chanlocs"][idx]["Z"] = float((-1) ** idx * (0.2 + 0.05 * idx))
    # Mark the last 2 channels (e.g. EOG) as having no coordinates
    for idx in (6, 7):
        eeg["chanlocs"][idx]["X"] = missing_coord
        eeg["chanlocs"][idx]["Y"] = missing_coord
        eeg["chanlocs"][idx]["Z"] = missing_coord

    kept, diag = clean_channels(
        eeg,
        subset_size=0.25,
        corr_threshold=-1.0,
        noise_threshold=100.0,
        num_samples=4,
        return_diagnostics=True,
    )
    assert kept["nbchan"] == 8
    np.testing.assert_array_equal(diag["usable_channels"], np.arange(6))


def test_half_or_fewer_located_channels_raises_location_error():
    """clean_channels.m raises when length([chanlocs.X/Y/Z]) is not greater than n*0.5."""
    eeg = _located_eeg(6, 800, 128.0)
    for idx in (3, 4, 5):
        eeg["chanlocs"][idx]["X"] = ""
        eeg["chanlocs"][idx]["Y"] = ""
        eeg["chanlocs"][idx]["Z"] = ""

    with pytest.raises(
        ValueError,
        match="most of your channels should have X,Y,Z location measurements",
    ):
        clean_channels(eeg, subset_size=0.25, num_samples=4)


def test_partial_coordinates_pass_the_length_gate_and_use_complete_channels():
    """Each coordinate can clear n*0.5 while complete XYZ triples do not; RANSAC keeps the triples."""
    eeg = _located_eeg(8, 800, 128.0)
    for idx in range(4):
        eeg["chanlocs"][idx]["Z"] = float((-1) ** idx * (0.2 + 0.05 * idx))
    original_x = eeg["chanlocs"][0]["X"]
    eeg["chanlocs"][0]["X"] = str(original_x)
    eeg["chanlocs"][4]["Y"] = ""
    eeg["chanlocs"][4]["Z"] = ""
    eeg["chanlocs"][5]["X"] = ""
    eeg["chanlocs"][5]["Z"] = []
    eeg["chanlocs"][6]["X"] = None
    eeg["chanlocs"][6]["Y"] = ()
    eeg["chanlocs"][7]["X"] = ""
    eeg["chanlocs"][7]["Y"] = ""
    eeg["chanlocs"][7]["Z"] = np.array([])

    kept, diag = clean_channels(
        eeg,
        subset_size=0.25,
        corr_threshold=-1.0,
        noise_threshold=100.0,
        num_samples=4,
        return_diagnostics=True,
    )
    assert kept["nbchan"] == 8
    np.testing.assert_array_equal(diag["usable_channels"], np.arange(4))
    np.testing.assert_allclose(diag["usable_locations"][0, 0], original_x)
