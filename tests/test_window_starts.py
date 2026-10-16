"""Window starts follow MATLAB's colon operator."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.miscfunc.misc import round_mat
from eegprep.plugins.clean_rawdata.asr_calibrate import asr_calibrate
from eegprep.plugins.clean_rawdata.clean_windows import clean_windows
from eegprep.plugins.clean_rawdata.private.windows import matlab_colon, matlab_window_offsets

_WINDOW_SAMPLES = (125, 128, 250, 256)
_OVERLAPS = (0.66, 0.5, 0.75)
_SWEEP = 3000


def _eeg(data: np.ndarray, srate: float) -> dict:
    samples = int(data.shape[1])
    return {
        "data": np.array(data, dtype=np.float64, copy=True),
        "srate": srate,
        "nbchan": int(data.shape[0]),
        "pnts": samples,
        "trials": 1,
        "xmin": 0.0,
        "xmax": (samples - 1) / srate,
        "event": [],
    }


def test_colon_matches_published_integer_and_endpoint_cases():
    np.testing.assert_array_equal(matlab_colon(1.0, 2.0, 9.0), [1.0, 3.0, 5.0, 7.0, 9.0])
    np.testing.assert_array_equal(matlab_colon(10.0, -2.0, 1.0), [10.0, 8.0, 6.0, 4.0, 2.0])
    tenths = matlab_colon(0.0, 0.1, 1.0)
    assert tenths.size == 11
    assert tenths[-1] == 1.0
    thirds = matlab_colon(0.0, 0.3, 1.0)
    assert thirds.size == 4
    assert thirds[-1] < 1.0


def test_reported_fractional_step_omits_the_extra_start():
    """N=256, overlap 0.66, S=1040. MATLAB starts here; 784 is not included."""
    offsets = matlab_window_offsets(1040, 256, 256 * (1.0 - 0.66))
    assert (offsets + 1).tolist() == [1, 88, 175, 262, 349, 436, 523, 610, 697]


def test_inexact_overlap_reflects_the_endpoint_before_rounding():
    """N=125, S=211. The reflected midpoint rounds to 44, not 43."""
    starts = np.asarray(round_mat(matlab_colon(1.0, 125 * (1.0 - 0.66), 211 - 125)), dtype=int)
    assert starts.tolist() == [1, 44, 86]


def test_fractional_overlap_keeps_samples_past_the_last_window():
    samples = 1040
    data = np.zeros((8, samples))
    data[:, 952:] = 1000.0
    _cleaned, mask = clean_windows(_eeg(data, 256.0), window_overlap=0.66)
    assert mask.shape == (samples,)
    assert mask.all()


def test_unrounded_window_length_stops_at_floor_n():
    """window_len*srate = 10.6. 0:N-1 has 10 samples, so index 75 stays outside every window."""
    samples = 80
    data = np.zeros((4, samples))
    data[:, 75:] = 1000.0
    _cleaned, mask = clean_windows(_eeg(data, 10.6), window_len=1.0, window_overlap=0.66)
    assert mask.all()


def _require_matlab(request) -> None:
    if request.config.getoption("--eeglab-backend") != "matlab":
        pytest.skip("requires --eeglab-backend=matlab")


def _asr_tolerance() -> tuple[float, float]:
    path = Path(__file__).resolve().parent / "fuzz" / "tolerances.toml"
    bound = tomllib.loads(path.read_text(encoding="utf-8"))["clean_asr"]
    return float(bound["abs"]), float(bound["rel"])


def _matlab_recording(data: np.ndarray, srate: float) -> dict:
    samples = int(data.shape[1])
    return {
        "data": np.array(data, dtype=np.float64, copy=True),
        "srate": float(srate),
        "nbchan": int(data.shape[0]),
        "pnts": float(samples),
        "trials": 1.0,
        "xmin": 0.0,
        "xmax": (samples - 1) / float(srate),
        "event": np.array([]),
        "urevent": np.array([]),
        "epoch": np.array([]),
        "etc": {},
    }


def test_matlab_window_starts_match_colon_sweep(request, eeglab_backend):
    _require_matlab(request)
    for window_samples in _WINDOW_SAMPLES:
        for overlap in _OVERLAPS:
            lengths = np.arange(window_samples + 1, window_samples + _SWEEP + 1, dtype=float)
            matlab = np.asarray(
                eeglab_backend("eegprep_test_window_starts", float(window_samples), float(overlap), lengths),
                dtype=float,
            )
            step = window_samples * (1.0 - overlap)
            for row, n_samples in enumerate(lengths.astype(int)):
                expected = matlab[row]
                expected = expected[np.isfinite(expected)].astype(int)
                actual = matlab_window_offsets(int(n_samples), window_samples, step) + 1
                if actual.shape != expected.shape or not np.array_equal(actual, expected):
                    raise AssertionError(
                        f"N={window_samples} overlap={overlap} S={n_samples}: {actual.tolist()} != {expected.tolist()}"
                    )


@pytest.mark.parametrize("samples", [5392, 5393])
def test_clean_windows_sample_mask_matches_matlab(request, eeglab_backend, samples: int):
    _require_matlab(request)
    rng = np.random.default_rng(490)
    data = rng.normal(scale=1.0, size=(8, samples))
    data[:, -800:] += 50.0
    recording = _matlab_recording(data, 256.0)
    shape = np.arange(1.7, 3.6, 0.15)
    _cleaned, python_mask = clean_windows(
        recording,
        max_bad_channels=0.2,
        zthresholds=(-3.5, 5.0),
        window_len=1.0,
        window_overlap=0.66,
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
        truncate_quant=(0.022, 0.6),
        step_sizes=(0.01, 0.01),
        shape_range=shape,
    )
    _signal, matlab_mask = eeglab_backend(
        "clean_windows",
        recording,
        0.2,
        np.array([-3.5, 5.0]),
        1.0,
        0.66,
        0.1,
        0.25,
        np.array([0.022, 0.6]),
        np.array([0.01, 0.01]),
        shape,
        nargout=2,
    )
    np.testing.assert_array_equal(
        np.asarray(python_mask, dtype=bool).ravel(),
        np.asarray(matlab_mask, dtype=bool).ravel(),
    )


def _align_threshold_rows(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip rows whose eigenvector sign differs. Squared thresholds ignore that sign."""
    aligned = np.array(candidate, dtype=float, copy=True)
    for index in range(aligned.shape[0]):
        if float(np.dot(aligned[index], reference[index])) < 0.0:
            aligned[index] *= -1.0
    return aligned


def test_asr_calibrate_fractional_step_matches_matlab(request, eeglab_backend):
    _require_matlab(request)
    samples = 1042
    rng = np.random.default_rng(490)
    time = np.arange(samples) / 256.0
    data = np.vstack([(1.0 + 0.02 * channel) * np.sin(2.0 * np.pi * (6.0 + channel) * time) for channel in range(8)])
    data += 0.05 * rng.standard_normal(data.shape)
    data[:, -180:] += 12.0
    data = np.asarray(data, dtype=np.float64)
    state = asr_calibrate(
        data,
        256.0,
        cutoff=5.0,
        blocksize=10,
        window_len=0.5,
        window_overlap=0.66,
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
        maxmem=64,
        compatibility="max",
    )
    matlab = eeglab_backend(
        "asr_calibrate",
        data,
        256.0,
        5.0,
        10.0,
        np.asarray(state["B"], dtype=float).ravel(),
        np.asarray(state["A"], dtype=float).ravel(),
        0.5,
        0.66,
        0.1,
        0.25,
        64.0,
    )
    atol, rtol = _asr_tolerance()
    np.testing.assert_allclose(
        np.real(np.asarray(state["M"], dtype=float)),
        np.real(np.asarray(matlab["M"], dtype=float)),
        atol=atol,
        rtol=rtol,
    )
    reference = np.real(np.asarray(matlab["T"], dtype=float))
    candidate = _align_threshold_rows(np.real(np.asarray(state["T"], dtype=float)), reference)
    np.testing.assert_allclose(candidate, reference, atol=atol, rtol=rtol)
