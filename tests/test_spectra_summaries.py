"""Headless spectrum summaries and the frozen spectopo numerical contract."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy.signal import get_window, welch
from scipy.stats import norm

from eegprep.functions.sigprocfunc.spectra import (
    channel_robust_amplitude,
    compute_spectra,
    line_noise_excess_db,
)
from eegprep.functions.sigprocfunc.spectopo import compute_spectra as spectopo_compute_spectra
from eegprep.functions.sigprocfunc.spectopo import spectopo


def _reference_signal():
    srate = 128.0
    pnts = 256
    times = np.arange(pnts) / srate
    data = np.vstack(
        [
            np.sin(2 * np.pi * 50 * times) + 0.25 * np.sin(2 * np.pi * 10 * times),
            0.5 * np.cos(2 * np.pi * 22 * times),
        ]
    )
    return data, pnts, srate


# Digests of spectopo's spectra and frequency grids for _reference_signal.
_FROZEN = {
    "default": (
        "43acecdc43b1b1c89306565ad8f55a9f91405e511460ced27cd7d64fa1f12d3b",
        "a2c297b161f5e61a7e16c45d19feb53ec8f2ae0103d1a97d7c91756ccfccba51",
        (2, 65),
        {},
    ),
    "percent": (
        "d14b6db8886cd0e1aa4be4626ef3de5fdcefce24bf856fce7207774a31f51c41",
        "a2c297b161f5e61a7e16c45d19feb53ec8f2ae0103d1a97d7c91756ccfccba51",
        (2, 65),
        {"percent": 50},
    ),
    "overlap": (
        "402138c32b745057b68a9e4e434c51536dd9a4694d146084ed3ff5eb30778393",
        "9287e14778ee197dd339ad57e60da907f264a23d5f074d4d284919838f48ded3",
        (2, 33),
        {"overlap": 32, "winsize": 64},
    ),
    "blackman": (
        "b311d13d3bf6fe5ab321433c06b58c07a009b9f2c6daf327aa33d709346d1353",
        "9287e14778ee197dd339ad57e60da907f264a23d5f074d4d284919838f48ded3",
        (2, 33),
        {"wintype": "blackmanharris"},
    ),
    "nfft": (
        "feb6c9d9049241199b388d5710517beb00098087508f92ebd87c472afd5cc928",
        "fd9a0ed0205cc3c1ac8223db9278715bb7349d63bf093fb76a18d502ac105b3b",
        (2, 129),
        {"nfft": 256, "winsize": 64},
    ),
    "epochs": (
        "6014d735b83bf8114f9b44ac4a3abfd9db26c191603908cd7c74d1972018fa64",
        "9287e14778ee197dd339ad57e60da907f264a23d5f074d4d284919838f48ded3",
        (2, 33),
        {},
    ),
    "concat": (
        "35185bca7611e8a9a383bb44b75c84a86ba911676b16296cb11aa132d485cb85",
        "9287e14778ee197dd339ad57e60da907f264a23d5f074d4d284919838f48ded3",
        (2, 33),
        {},
    ),
}


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def test_spectopo_numeric_output_matches_the_moved_estimator():
    data, pnts, srate = _reference_signal()
    epochs = data.reshape(2, 64, 4)
    inputs = {"epochs": (epochs, 64), "concat": (data, 64)}
    for name, (spectra_hash, frequency_hash, shape, kwargs) in _FROZEN.items():
        array, frames = inputs.get(name, (data, pnts))
        spectra, freqs, first, second, specstd, figure = spectopo(array, frames, srate, plot="off", **kwargs)
        direct, direct_freqs, direct_std = compute_spectra(array, frames, srate, **kwargs)
        assert spectra.shape == shape
        assert _digest(spectra) == spectra_hash
        assert _digest(freqs) == frequency_hash
        assert np.array_equal(spectra, direct)
        assert np.array_equal(freqs, direct_freqs)
        assert spectopo_compute_spectra is compute_spectra
        assert specstd is None and direct_std is None
        assert first is None and second is None and figure is None


def test_default_grid_is_density_sampled_at_1_hz_through_nyquist():
    srate = 250.0
    pnts = 1000
    times = np.arange(pnts) / srate
    data = np.vstack([np.sin(2 * np.pi * 50 * times), 0.25 * np.sin(2 * np.pi * 10 * times)])
    spectra, freqs, _std = compute_spectra(data, pnts, srate)
    assert float(freqs[1] - freqs[0]) == 1.0
    assert float(freqs[0]) == 0.0
    assert float(freqs[-1]) == srate / 2.0
    assert np.all(freqs <= srate / 2.0)
    selected = (freqs >= 1.0) & (freqs <= 70.0)
    assert np.array_equal(freqs[selected], np.arange(1, 71, dtype=float))

    nperseg = 64
    coarse, coarse_freqs, _std = compute_spectra(data[:, :256], 256, 128.0, winsize=nperseg, nfft=nperseg)
    window = get_window("hamming", nperseg, fftbins=False)
    density_freqs, density = welch(
        data[:, :256],
        fs=128.0,
        window=window,
        nperseg=nperseg,
        noverlap=0,
        nfft=nperseg,
        axis=1,
        detrend=False,
        scaling="density",
    )
    _spectrum_freqs, spectrum = welch(
        data[:, :256],
        fs=128.0,
        window=window,
        nperseg=nperseg,
        noverlap=0,
        nfft=nperseg,
        axis=1,
        detrend=False,
        scaling="spectrum",
    )
    density_db = 10.0 * np.log10(np.maximum(density, np.finfo(float).tiny))
    spectrum_db = 10.0 * np.log10(np.maximum(spectrum, np.finfo(float).tiny))
    assert float(coarse_freqs[1] - coarse_freqs[0]) == 2.0
    assert np.array_equal(coarse_freqs, density_freqs)
    assert np.array_equal(coarse, density_db)
    assert not np.allclose(coarse, spectrum_db, rtol=0, atol=1e-6)
    assert float(coarse_freqs[-1]) <= 64.0


def test_line_noise_excess_is_the_db_mean_of_neighboring_bins():
    srate = 250.0
    pnts = 500
    times = np.arange(pnts) / srate
    data = np.vstack(
        [
            np.sin(2 * np.pi * 50 * times) + 0.2 * np.sin(2 * np.pi * 10 * times),
            0.5 * np.sin(2 * np.pi * 60 * times),
        ]
    )
    eeg = {"data": data, "pnts": pnts, "srate": srate}
    spectra, freqs, _std = compute_spectra(data, pnts, srate)
    excess = line_noise_excess_db(eeg, freqs=(60, 50), neighbours=2)
    assert excess.shape == (2, 2)
    for column, frequency in enumerate((60, 50)):
        index = int(np.flatnonzero(np.abs(freqs - frequency) <= 1e-6)[0])
        neighbors = np.concatenate((spectra[:, index - 2 : index], spectra[:, index + 1 : index + 3]), axis=1)
        decibel_baseline = neighbors.mean(axis=1)
        linear = 10.0 ** (spectra / 10.0)
        linear_neighbors = np.concatenate((linear[:, index - 2 : index], linear[:, index + 1 : index + 3]), axis=1)
        linear_baseline = 10.0 * np.log10(linear_neighbors.mean(axis=1))
        assert np.array_equal(excess[:, column], spectra[:, index] - decibel_baseline)
        assert np.max(np.abs(decibel_baseline - linear_baseline)) > 0.1

    one_sided = line_noise_excess_db(eeg, freqs=50, neighbours=2)
    assert one_sided.shape == (2, 1)
    assert np.array_equal(one_sided[:, 0], excess[:, 1])


def test_line_noise_excess_pools_epochs_in_linear_power_and_column_major_order():
    srate = 250.0
    pnts = 250
    times = np.arange(pnts) / srate
    epoch0 = np.vstack([np.sin(2 * np.pi * 50 * times), np.cos(2 * np.pi * 60 * times)])
    epoch1 = np.vstack([0.1 * np.sin(2 * np.pi * 50 * times), 2 * np.cos(2 * np.pi * 60 * times)])
    epochs = np.stack((epoch0, epoch1), axis=-1)
    concatenated = np.concatenate((epoch0, epoch1), axis=1)
    interleaved = np.reshape(epochs, (2, pnts * 2), order="C")
    kwargs = {"freqs": (50, 60), "neighbours": 1}
    epoched = line_noise_excess_db({"data": epochs, "pnts": pnts, "srate": srate}, **kwargs)
    column_major = line_noise_excess_db({"data": concatenated, "pnts": pnts, "srate": srate}, **kwargs)
    row_major = line_noise_excess_db({"data": interleaved, "pnts": pnts, "srate": srate}, **kwargs)
    assert np.array_equal(epoched, column_major)
    assert not np.array_equal(epoched, row_major)


def test_line_noise_excess_rejects_unmeasurable_frequencies():
    long = {"data": np.zeros((1, 200)), "pnts": 200, "srate": 100.0}
    with pytest.raises(ValueError, match="above the Nyquist frequency"):
        line_noise_excess_db(long, freqs=(60, 50))
    with pytest.raises(ValueError, match="neighboring bins"):
        line_noise_excess_db(long, freqs=50)
    short = {"data": np.zeros((1, 20)), "pnts": 20, "srate": 250.0}
    with pytest.raises(ValueError, match="not a bin"):
        line_noise_excess_db(short, freqs=60)
    measured = line_noise_excess_db(short, freqs=50)
    assert measured.shape == (1, 1)
    with pytest.raises(ValueError, match="above the Nyquist frequency"):
        line_noise_excess_db(long, freqs=50 + 2e-6)
    with pytest.raises(ValueError, match="not a bin"):
        line_noise_excess_db(long, freqs=10.5)
    within = line_noise_excess_db(long, freqs=(10, 10 + 5e-7))
    assert np.array_equal(within[:, 0], within[:, 1])


def test_line_noise_excess_rejects_bad_arguments_without_changing_the_dataset():
    data = np.sin(2 * np.pi * 10 * np.arange(100) / 100.0).reshape(1, 100).astype(np.float32)
    original = data.copy()
    eeg = {"data": data, "pnts": 100, "srate": 100.0, "setname": "keep"}
    excess = line_noise_excess_db(eeg, freqs=10)
    assert excess.shape == (1, 1)
    assert eeg["data"] is data
    assert data.dtype == np.float32
    assert np.array_equal(data, original)
    assert eeg["setname"] == "keep"
    with pytest.raises(ValueError, match="neighbours"):
        line_noise_excess_db(eeg, neighbours=0)
    with pytest.raises(ValueError, match="neighbours"):
        line_noise_excess_db(eeg, neighbours=1.5)
    with pytest.raises(ValueError, match="neighbours"):
        line_noise_excess_db(eeg, neighbours=True)
    with pytest.raises(ValueError, match="finite"):
        line_noise_excess_db(eeg, freqs=(10, np.nan))
    with pytest.raises(TypeError, match="dataset dictionary"):
        line_noise_excess_db(data, freqs=10)
    with pytest.raises(ValueError, match="srate"):
        line_noise_excess_db({"data": data}, freqs=10)
    with pytest.raises(ValueError, match="positive finite"):
        line_noise_excess_db({"data": data, "srate": 0}, freqs=10)


def test_channel_robust_amplitude_is_pooled_normal_scaled_mad():
    epochs = np.zeros((2, 3, 2))
    epochs[0, :, 0] = [1, 2, 3]
    epochs[0, :, 1] = [10, 11, 12]
    epochs[1, :, 0] = [0, 0, 0]
    epochs[1, :, 1] = [4, 5, 6]
    pooled = epochs.reshape(2, -1)
    center = np.median(pooled, axis=1)
    raw = np.median(np.abs(pooled - center[:, None]), axis=1)
    scale = 1.0 / norm.ppf(0.75)
    expected = raw * scale
    per_epoch = np.array(
        [
            np.mean([_scaled_mad(epochs[0, :, 0]), _scaled_mad(epochs[0, :, 1])]),
            np.mean([_scaled_mad(epochs[1, :, 0]), _scaled_mad(epochs[1, :, 1])]),
        ]
    )
    amplitude = channel_robust_amplitude({"data": epochs})
    assert amplitude.shape == (2,)
    assert np.array_equal(amplitude, expected)
    assert not np.allclose(amplitude, per_epoch)

    noisy = np.array([[1.0, 2.0, np.nan, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 100.0]])
    nan_amplitude = channel_robust_amplitude({"data": noisy})
    assert np.isnan(nan_amplitude[0])
    assert nan_amplitude[1] == _scaled_mad(noisy[1])

    stored = np.array([[1, 2, 3, 4, 100], [0, 0, 0, 0, 1]], dtype=np.float32)
    before = stored.copy()
    dataset = {"data": stored}
    channel_robust_amplitude(dataset)
    assert dataset["data"] is stored
    assert stored.dtype == np.float32
    assert np.array_equal(stored, before)


def test_numerical_imports_do_not_load_matplotlib():
    code = textwrap.dedent(
        """
        import sys
        import eegprep.functions.sigprocfunc.spectra as spectra
        from eegprep import channel_robust_amplitude, compute_spectra, line_noise_excess_db

        assert "matplotlib" not in sys.modules
        data = __import__("numpy").sin(__import__("numpy").arange(100) / 10).reshape(1, 100)
        eeg = {"data": data, "pnts": 100, "srate": 100.0}
        compute_spectra(data, 100, 100.0)
        line_noise_excess_db(eeg, freqs=10)
        channel_robust_amplitude(eeg)
        assert spectra.compute_spectra is compute_spectra
        assert "matplotlib" not in sys.modules
        from eegprep.functions.sigprocfunc import cbar

        assert callable(cbar)
        assert "matplotlib" in sys.modules
        print("ok")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_package_root_exports_the_numerical_functions():
    import eegprep

    from eegprep.functions.sigprocfunc.spectra import channel_robust_amplitude as amplitude
    from eegprep.functions.sigprocfunc.spectra import compute_spectra as spectra
    from eegprep.functions.sigprocfunc.spectra import line_noise_excess_db as excess
    from eegprep.functions.sigprocfunc.spectopo import spectopo as plot_spectopo

    assert eegprep.compute_spectra is spectra
    assert eegprep.line_noise_excess_db is excess
    assert eegprep.channel_robust_amplitude is amplitude
    assert eegprep.spectopo is plot_spectopo


def _scaled_mad(samples: np.ndarray) -> float:
    center = float(np.median(samples))
    return float(np.median(np.abs(np.asarray(samples, dtype=float) - center)) / norm.ppf(0.75))
