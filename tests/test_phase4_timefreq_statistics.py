from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import os
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.io
from scipy import stats

from tests.eeglab_tests import assert_matlab_near, eeglab_test
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_eventstat import event_values, pop_eventstat
from eegprep.functions.popfunc.pop_crossf import pop_crossf
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_newcrossf import pop_newcrossf
from eegprep.functions.popfunc.pop_newtimef import pop_newtimef
from eegprep.functions.popfunc.pop_timef import pop_timef
from eegprep.functions.popfunc.pop_signalstat import pop_signalstat
from eegprep.functions.sigprocfunc.signalstat import signalstat
from eegprep.functions.guifunc.tf_cycle_calc_dialog import tf_cycle_calc_dialog_spec
from eegprep.functions.timefreqfunc.angtimewarp import angtimewarp
from eegprep.functions.timefreqfunc._bootstrap import (
    bootstrap_indices,
    resample_trials,
    threshold_vector,
    thresholds_by_frequency,
)
from eegprep.functions.timefreqfunc.bootstat import bootstat, bootstrap_threshold, exact_p_values
from eegprep.functions.timefreqfunc.correctfit import correctfit
from eegprep.functions.timefreqfunc.dftfilt2 import dftfilt2
from eegprep.functions.timefreqfunc.dftfilt3 import dftfilt3
from eegprep.functions.statistics.stat_surrogate_pvals import stat_surrogate_pvals
from eegprep.functions.timefreqfunc.newcrossf import _upper_thresholds_by_frequency
from eegprep.functions.timefreqfunc.newcrossf import newcrossf
from eegprep.functions.timefreqfunc._pac_support import _empirical_pvalue as pac_empirical_pvalue
from eegprep.functions.timefreqfunc.newtimef import (
    _baseline_pvalues,
    _bootstrap_itc,
    _bootstrap_power,
    _thresholds_by_frequency,
)
from eegprep.functions.timefreqfunc.newtimef import newtimef
from eegprep.functions.timefreqfunc.newtimefbaseln import newtimefbaseln
from eegprep.functions.timefreqfunc.rsadjust import rsadjust
from eegprep.functions.timefreqfunc.rsfit import rsfit
from eegprep.functions.timefreqfunc.rsget import rsget
from eegprep.functions.timefreqfunc.rspdfsolv import rspdfsolv
from eegprep.functions.timefreqfunc.rspfunc import rspfunc
from eegprep.functions.timefreqfunc.tf_cycle_calc import tf_cycle_calc
from eegprep.functions.timefreqfunc.timefreq import _replace_zero_bins, timefreq
from eegprep.functions.timefreqfunc.timewarp import timewarp
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg_with_ica


@pytest.fixture(scope="module")
def sample_eeg():
    return pop_loadset(SAMPLE_DATASET_PATH)


@pytest.fixture(scope="module")
def sample_epoch(sample_eeg):
    eeg, _command = pop_epoch(deepcopy(sample_eeg), ["square"], [-0.1, 0.2], return_com=True)
    return eeg


@pytest.fixture
def ica_epoch():
    return create_test_eeg_with_ica(n_channels=6, n_samples=96, n_trials=5, n_components=4)


@pytest.mark.gui
@eeglab_test("unittesting_sigprocfunc/newtimef/sigprocfunc_newtimef_wrapperTest.m", "test_test_newtimef")
def test_reference_newtimef_frequency_plots(eeglab_backend, request):
    srate, frames = 1000.0, 600.0
    times = np.arange(frames)[None, :] / srate
    data = 0.7 * np.sin(2 * np.pi * 25 * times) + np.cos(2 * np.pi * 50 * times)
    for cycles, frequencies in ((0.0, [[3.0, 100.0]]), (5.0, [[10.0, 100.0]])):
        if request.config.getoption("--eeglab-backend") == "matlab":
            request.getfixturevalue("eeglab_matlab_engine").figure(nargout=0)
        else:
            plt.figure()
        eeglab_backend(
            "newtimef",
            data,
            frames,
            np.array([[0.0, frames / srate * 1000]]),
            srate,
            cycles,
            "winsize",
            200.0,
            "freqs",
            np.array(frequencies),
            "baseline",
            np.nan,
            nargout=0,
        )
        if request.config.getoption("--eeglab-backend") == "matlab":
            request.getfixturevalue("eeglab_matlab_engine").close(nargout=0)
        else:
            plt.close()
    spectrum = np.fft.fft(data[:, :402], axis=1)
    _frequencies = np.linspace(0, srate / 2, spectrum.shape[1] // 2)[1:]
    _spectrum = spectrum[:, 1 : spectrum.shape[1] // 2]


@eeglab_test("unittesting_sigprocfunc/newtimef/sigprocfunc_newtimef_wrapperTest.m", "test_test_newtimef2")
def test_reference_newtimef_original_baseline_formulas(eeglab_backend):
    data = np.random.default_rng().random((100, 10))
    options = (
        "timesout",
        np.arange(20.0, 81.0, 10.0)[None, :] * 10,
        "padratio",
        1.0,
        "winsize",
        32.0,
        "plotitc",
        "off",
        "plotersp",
        "off",
        "verbose",
        "off",
        "outputformat",
        "plot",
    )
    # Literal timefreqfft/hanning formulas from the local MATLAB source helpers.
    half_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, 17) / 33))
    window = np.concatenate((half_window, half_window[::-1]))[:, None]
    transforms = []
    for center in (21, 31, 41):
        segment = data[center - 16 : center + 16]
        segment = segment - segment.mean(axis=0)
        transforms.append(np.fft.fft(segment * window, 32, axis=0)[1:17] * 2 / 0.375 / 32)
    first, second, third = transforms
    power1, power2, power3 = [np.mean(np.abs(values) ** 2, axis=1, keepdims=True) for values in transforms]
    frequencies = np.linspace(0, 50, 17)[None, 1:]
    wavelet_frequencies = np.array([[10.0, 12.0, 14.0, 16.0, 18.0]])
    wavelets = eeglab_backend("dftfilt3", wavelet_frequencies, 3.0, 100.0, "cycleinc", "linear")
    tapered = []
    for wavelet in np.asarray(wavelets, dtype=object).ravel():
        wavelet = np.asarray(wavelet).ravel()
        half = (wavelet.size - 1) // 2
        segment = data[np.arange(-half, half + 1) + 30]
        tapered.append(np.sum((segment - segment.mean(axis=0)) * wavelet[:, None], axis=0))
    wavelet_power = np.mean(np.abs(tapered) ** 2, axis=1, keepdims=True)
    itc_expected = np.abs(np.mean(first / np.abs(first), axis=1, keepdims=True))
    baseline_ratio = power2 / power1
    trial_ratio = np.mean(np.abs(second / np.abs(first)) ** 2, axis=1, keepdims=True)
    mean_base = (power1 + power2) / 2
    std_base = np.sqrt((power1 - mean_base) ** 2 + (power2 - mean_base) ** 2)
    normalized = (power3 - mean_base) / std_base
    trial_mean = np.abs(first) ** 2 / 2 + np.abs(second) ** 2 / 2
    trial_std = np.sqrt((np.abs(first) ** 2 - trial_mean) ** 2 + (np.abs(second) ** 2 - trial_mean) ** 2)
    trial_normalized = np.mean((np.abs(third) ** 2 - trial_mean) / trial_std, axis=1, keepdims=True)
    arguments = (data, 100.0, np.array([[0.0, 990.0]]), 100.0, 0.0)
    absolute, itc, _, _, actual_frequencies = eeglab_backend(
        "newtimef",
        *arguments,
        "baseline",
        np.nan,
        "scale",
        "abs",
        *options,
        nargout=5,
    )
    logarithmic = eeglab_backend("newtimef", *arguments, "baseline", np.nan, "scale", "log", *options)
    baseline_abs, _, powbase_abs = eeglab_backend(
        "newtimef",
        *arguments,
        "baseline",
        250.0,
        "scale",
        "abs",
        *options,
        nargout=3,
    )
    baseline_log, _, powbase_log = eeglab_backend(
        "newtimef",
        *arguments,
        "baseline",
        250.0,
        "scale",
        "log",
        *options,
        nargout=3,
    )
    trial_abs = eeglab_backend("newtimef", *arguments, "baseline", 250.0, "scale", "abs", "trialbase", "on", *options)
    trial_log = eeglab_backend("newtimef", *arguments, "baseline", 250.0, "scale", "log", "trialbase", "on", *options)
    normalized_abs = eeglab_backend(
        "newtimef", *arguments, "baseline", 350.0, "scale", "abs", "basenorm", "on", *options
    )
    normalized_trial = eeglab_backend(
        "newtimef",
        *arguments,
        "baseline",
        350.0,
        "scale",
        "abs",
        "trialbase",
        "on",
        "basenorm",
        "on",
        *options,
    )
    wavelet_abs = eeglab_backend(
        "newtimef",
        data,
        100.0,
        np.array([[0.0, 990.0]]),
        100.0,
        3.0,
        "baseline",
        np.nan,
        "scale",
        "abs",
        "freqs",
        wavelet_frequencies,
        *options,
    )
    for expected, actual in (
        (frequencies, actual_frequencies),
        (itc_expected, np.abs(itc[:, :1])),
        (power1, absolute[:, :1]),
        (10 * np.log10(power1), logarithmic[:, :1]),
        (baseline_ratio, baseline_abs[:, 1:2]),
        (power1, powbase_abs.T),
        (10 * np.log10(baseline_ratio), baseline_log[:, 1:2]),
        (10 * np.log10(power1), powbase_log.T),
        (trial_ratio, trial_abs[:, 1:2]),
        (10 * np.log10(trial_ratio), trial_log[:, 1:2]),
        (normalized, normalized_abs[:, 2:3]),
        (trial_normalized, normalized_trial[:, 2:3]),
        (wavelet_power, wavelet_abs[:, 1:2]),
    ):
        assert np.all(np.abs(expected - actual) < 1e-8)


def test_newtimef_upstream_fft_wavelet_and_baseline_formulas_match_exactly():
    data = np.random.RandomState(0).rand(100, 10)
    output_times = np.arange(20, 81, 10) * 10
    common = dict(
        timesout=output_times,
        padratio=1,
        winsize=32,
        plotitc="off",
        plotersp="off",
        verbose="off",
        outputformat="plot",
        plot="off",
    )
    window = np.hanning(34)[1:-1]

    def fft_window(center):
        segment = data[center - 16 : center + 16]
        segment = segment - segment.mean(axis=0)
        return np.fft.fft(segment * window[:, None], axis=0)[1:17] * 2 / 0.375 / 32

    fft_data = [fft_window(center) for center in (21, 31, 41)]
    power = [np.mean(np.abs(values) ** 2, axis=1) for values in fft_data]

    absolute = newtimef(data, 100, [0, 990], 100, 0, baseline=np.nan, scale="abs", **common)
    logarithmic = newtimef(data, 100, [0, 990], 100, 0, baseline=np.nan, scale="log", **common)
    np.testing.assert_allclose(absolute.ersp[:, 0], power[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(logarithmic.ersp[:, 0], 10 * np.log10(power[0]), rtol=0, atol=1e-13)
    expected_itc = np.mean(fft_data[0] / np.abs(fft_data[0]), axis=1)
    np.testing.assert_allclose(np.abs(absolute.itc[:, 0]), np.abs(expected_itc), rtol=0, atol=1e-14)

    baseline_abs = newtimef(data, 100, [0, 990], 100, 0, baseline=250, scale="abs", **common)
    baseline_log = newtimef(data, 100, [0, 990], 100, 0, baseline=250, scale="log", **common)
    np.testing.assert_allclose(baseline_abs.ersp[:, 1], power[1] / power[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(baseline_abs.powbase, power[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(baseline_log.ersp[:, 1], 10 * np.log10(power[1] / power[0]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(baseline_log.powbase, 10 * np.log10(power[0]), rtol=0, atol=1e-13)

    trial_abs = newtimef(data, 100, [0, 990], 100, 0, baseline=250, scale="abs", trialbase="on", **common)
    trial_log = newtimef(data, 100, [0, 990], 100, 0, baseline=250, scale="log", trialbase="on", **common)
    trial_ratio = np.mean(np.abs(fft_data[1] / fft_data[0]) ** 2, axis=1)
    np.testing.assert_allclose(trial_abs.ersp[:, 1], trial_ratio, rtol=0, atol=1e-12)
    np.testing.assert_allclose(trial_log.ersp[:, 1], 10 * np.log10(trial_ratio), rtol=0, atol=1e-12)

    mean_base = (power[0] + power[1]) / 2
    std_base = np.sqrt((power[0] - mean_base) ** 2 + (power[1] - mean_base) ** 2)
    normalized = newtimef(data, 100, [0, 990], 100, 0, baseline=350, scale="abs", basenorm="on", **common)
    np.testing.assert_allclose(normalized.ersp[:, 2], (power[2] - mean_base) / std_base, rtol=0, atol=1e-11)

    trial_mean = np.abs(fft_data[0]) ** 2 / 2 + np.abs(fft_data[1]) ** 2 / 2
    trial_std = np.sqrt((np.abs(fft_data[0]) ** 2 - trial_mean) ** 2 + (np.abs(fft_data[1]) ** 2 - trial_mean) ** 2)
    trial_expected = np.mean((np.abs(fft_data[2]) ** 2 - trial_mean) / trial_std, axis=1)
    trial_normalized = newtimef(
        data,
        100,
        [0, 990],
        100,
        0,
        baseline=350,
        scale="abs",
        trialbase="on",
        basenorm="on",
        **common,
    )
    np.testing.assert_allclose(trial_normalized.ersp[:, 2], trial_expected, rtol=0, atol=1e-11)

    frequencies = [10, 12, 14, 16, 18]
    wavelets, *_ = dftfilt3(frequencies, 3, 100, cycleinc="linear")
    wavelet_values = []
    for wavelet in wavelets:
        half = (len(wavelet) - 1) // 2
        segment = data[np.arange(-half, half + 1) + 30]
        segment = segment - segment.mean(axis=0)
        wavelet_values.append(np.sum(segment * wavelet[:, None], axis=0))
    wavelet_power = np.mean(np.abs(wavelet_values) ** 2, axis=1)
    wavelet_result = newtimef(data, 100, [0, 990], 100, 3, baseline=np.nan, scale="abs", freqs=frequencies, **common)
    np.testing.assert_allclose(wavelet_result.ersp[:, 1], wavelet_power, rtol=0, atol=1e-14)


def test_newtimef_defaults_frequency_range_to_eeglab_maxfreq():
    # With no explicit freqs, EEGLAB stops at maxfreq=50 (capped at Nyquist), not the full band.
    times = np.arange(256) / 128
    trials = np.stack([np.sin(2 * np.pi * 10 * times + phase) for phase in (0.0, 0.2, 0.5)], axis=1)

    result = newtimef(trials, 256, [0, 2000], 128, [3, 0.5], plot="off")

    assert result.freqs.max() <= 50.0 + 1e-9
    assert result.freqs.max() > 30.0  # the band is not truncated too aggressively


def test_newtimef_freqrange_alias_freqscale_and_scale_validation():
    # freqrange aliases freqs; freqscale='log' spaces the output frequencies geometrically;
    # an unknown scale fails fast.
    srate = 128
    trials = _oscillation_trials(srate, 256, [0.0, 0.3, 0.6])

    aliased = newtimef(trials, 256, [0, 2000], srate, [3, 0.5], freqrange=[6, 30], nfreqs=6, plot="off")
    assert aliased.freqs.min() == pytest.approx(6.0)
    assert aliased.freqs.max() == pytest.approx(30.0)

    log_freqs = newtimef(
        trials, 256, [0, 2000], srate, [3, 0.5], freqs=[6, 30], nfreqs=6, freqscale="log", plot="off"
    ).freqs
    ratios = log_freqs[1:] / log_freqs[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)  # constant ratio -> geometric spacing

    with pytest.raises(ValueError, match="scale"):
        newtimef(trials, 256, [0, 2000], srate, [3, 0.5], scale="bogus", plot="off")


def test_newtimef_supplied_powbase_shifts_ersp_by_db_offset():
    # A supplied baseline spectrum (dB) sets the log-power baseline directly; raising it by
    # K dB lowers the whole ERSP by K dB (EEGLAB log-subtracts the supplied powbase).
    srate = 128
    trials = _oscillation_trials(srate, 256, [0.0, 0.3, 0.6])
    common = dict(freqs=[6, 20], nfreqs=6, plot="off")

    nfreq = newtimef(trials, 256, [0, 2000], srate, [3, 0.5], **common).freqs.size
    base0 = newtimef(trials, 256, [0, 2000], srate, [3, 0.5], powbase=np.zeros(nfreq), **common)
    base3 = newtimef(trials, 256, [0, 2000], srate, [3, 0.5], powbase=np.full(nfreq, 3.0), **common)
    np.testing.assert_allclose(base3.ersp, base0.ersp - 3.0, rtol=1e-6, atol=1e-6)
    # the supplied dB spectrum round-trips (dB -> linear on input, linear -> dB on return)
    np.testing.assert_allclose(base0.powbase, np.zeros(nfreq), atol=1e-9)
    np.testing.assert_allclose(base3.powbase, np.full(nfreq, 3.0), rtol=1e-9)


def test_newtimef_baseline_forms_control_powbase_units():
    # The baseline guard must accept every EEGLAB baseline form. A multi-window (nested-list)
    # baseline and an empty-list baseline are both enabled, so log-scale powbase comes back in dB
    # (= 10*log10 of the absolute-scale powbase); a NaN baseline is disabled, so powbase stays in
    # absolute power. Regression guard: a nested-list baseline previously crashed the guard.
    srate = 128
    trials = _oscillation_trials(srate, 256, [0.0, 0.3, 0.6])

    def powbase(baseline, scale="log"):
        result = newtimef(
            trials,
            256,
            [-1000, 1000],
            srate,
            [3, 0.5],
            freqs=[6, 20],
            nfreqs=6,
            baseline=baseline,
            scale=scale,
            plot="off",
        )
        return np.asarray(result.powbase).ravel()

    for baseline in ([[-400, -200], [200, 400]], []):  # both enabled -> dB
        log_db = powbase(baseline)
        assert np.isfinite(log_db).all()
        np.testing.assert_allclose(log_db, 10.0 * np.log10(powbase(baseline, scale="abs")), rtol=1e-6, atol=1e-6)

    # NaN disables the baseline, so log-scale powbase stays in absolute power (as in EEGLAB)
    np.testing.assert_allclose(powbase(np.nan), powbase(np.nan, scale="abs"), rtol=1e-6, atol=1e-6)


def test_newtimef_supplied_1d_bootstrap_thresholds_flag_extremes():
    # A 1-D erspboot supplies a symmetric per-frequency band and a 1-D itcboot an upper magnitude
    # threshold; supplied limits bypass the bootstrap and mask by comparison. (The dB / symmetric-band
    # reading of a 1-D vector is an EEGPrep convention; EEGLAB documents pboot as an (nfreqs, 2) array.)
    srate = 128
    trials = _oscillation_trials(srate, 256, [0.0, 0.3, 0.6])
    common = dict(freqs=[6, 20], nfreqs=6, plot="off")

    base = newtimef(trials, 256, [0, 2000], srate, [3, 0.5], **common)
    # per-frequency thresholds that straddle the data, so the masks are genuinely mixed
    ersp_thresh = np.percentile(np.abs(base.ersp), 60, axis=1)
    itc_thresh = np.percentile(np.abs(base.itc), 40, axis=1)
    result = newtimef(
        trials, 256, [0, 2000], srate, [3, 0.5], alpha=0.05, erspboot=ersp_thresh, itcboot=itc_thresh, **common
    )
    np.testing.assert_array_equal(result.ersp_significant, np.abs(result.ersp) >= ersp_thresh[:, None])
    np.testing.assert_array_equal(result.itc_significant, np.abs(result.itc) >= itc_thresh[:, None])
    assert 0.0 < result.ersp_significant.mean() < 1.0  # mixed mask, not all-False/all-True
    assert 0.0 < result.itc_significant.mean() < 1.0


# --- timefreq numeric-parity regression guards (EEGLAB timefreq.m) ----------


def _oscillation_trials(srate, frames, phases):
    times = np.arange(frames) / srate
    return np.stack([np.sin(2 * np.pi * 10 * times + phase) for phase in phases], axis=1)


def test_timefreq_fft_path_ignores_detrend():
    # EEGLAB's active FFT branch only subtracts the window mean; 'detrend' is a
    # no-op for cycles=0 (timefreq.m 323-346), even with a linear trend present.
    srate = 128.0
    frames = 128
    trend = np.linspace(0.0, 3.0, frames)[:, None] * np.asarray([1.0, 0.7, 1.3])
    trials = _oscillation_trials(srate, frames, [0.0, 0.2, 0.5]) + trend
    common = dict(frames=frames, cycles=0, tlimits=[0, 1000], freqs=[5, 20], ntimesout=12, padratio=2)
    off = timefreq(trials, srate, detrend="off", **common)
    on = timefreq(trials, srate, detrend="on", **common)
    np.testing.assert_allclose(on.tfdata, off.tfdata, rtol=1e-12, atol=1e-12)


def test_timefreq_subitc_returns_presubtraction_itc():
    # EEGLAB returns the pre-subtraction ITC and only then subtracts it from the
    # single-trial estimates (timefreq.m 552-561).
    srate = 128.0
    frames = 128
    trials = _oscillation_trials(srate, frames, [0.0, 0.2, 0.5])
    common = dict(frames=frames, cycles=0, tlimits=[0, 1000], freqs=[5, 20], ntimesout=12, padratio=2)
    off = timefreq(trials, srate, subitc="off", **common)
    on = timefreq(trials, srate, subitc="on", **common)
    np.testing.assert_allclose(on.itcvals, off.itcvals, rtol=1e-12, atol=1e-12)
    assert not np.allclose(on.tfdata, off.tfdata)


def test_timefreq_keeps_one_output_bin_per_requested_frequency():
    # EEGLAB maps each requested frequency to its nearest computed bin without
    # de-duplicating (timefreq.m 565-576); a coarse grid still yields one output
    # frequency per request, duplicates included.
    srate = 128.0
    frames = 128
    trials = _oscillation_trials(srate, frames, [0.0, 0.2])
    requested = [5, 6, 7, 8, 9, 10]
    result = timefreq(
        trials,
        srate,
        frames=frames,
        cycles=0,
        tlimits=[0, 1000],
        freqs=requested,
        winsize=16,
        padratio=2,
        ntimesout=12,
    )
    assert result.freqs.size == len(requested)
    assert result.tfdata.shape[0] == len(requested)
    assert np.unique(result.freqs).size < result.freqs.size


def test_timefreq_output_times_round_half_away_from_zero():
    # Output windows are centered via eeg_lat2point + round-half-away-from-zero
    # (timefreq.m 657). A request landing exactly between two samples rounds up,
    # where a nearest-sample argmin would fall back to the lower index.
    srate = 100.0
    frames = 101
    tlimits = [0.0, 1000.0]  # 10 ms sample spacing
    winsize = 20
    trials = _oscillation_trials(srate, frames, [0.0, 0.2])
    result = timefreq(
        trials,
        srate,
        frames=frames,
        cycles=0,
        tlimits=tlimits,
        freqs=[5, 20],
        winsize=winsize,
        padratio=2,
        timesout=[205, 405, 605],
    )
    # Each request sits exactly between samples (e.g. 405 ms between 400 and 410);
    # EEGLAB rounds up, so the selected sample times are the upper neighbours.
    np.testing.assert_allclose(result.times, [210.0, 410.0, 610.0], rtol=1e-12, atol=1e-12)


def test_timefreq_subsample_times_match_eeglab_colon():
    # Negative ntimesout subsamples on EEGLAB's colon grid (timefreq.m 629); the
    # inclusive upper bound is frames-ceil(winsize/2)-1, with no extra tail point.
    srate = 128.0
    frames = 121
    tlimits = [0.0, 1000.0]
    winsize = 16
    nsub = 10
    trials = _oscillation_trials(srate, frames, [0.0, 0.2])
    result = timefreq(
        trials,
        srate,
        frames=frames,
        cycles=0,
        tlimits=tlimits,
        freqs=[5, 20],
        winsize=winsize,
        padratio=2,
        ntimesout=-nsub,
    )
    start1 = int(np.ceil(winsize / 2 + nsub / 2))
    stop1 = frames - int(np.ceil(winsize / 2)) - 1
    expected_idx = np.arange(start1, stop1 + 1, nsub) - 1
    timevect = np.linspace(tlimits[0], tlimits[1], frames)
    np.testing.assert_allclose(result.times, timevect[expected_idx], rtol=1e-12, atol=1e-12)


def test_timefreq_wavelet_detrend_only_applies_to_single_channel():
    # EEGLAB detrends only the single-channel wavelet branch (timefreq.m 457-462);
    # the multichannel branch subtracts the window mean only.
    srate = 128.0
    frames = 128
    tlimits = [0.0, 1000.0]
    trend = np.linspace(0.0, 4.0, frames)[:, None]
    ch0 = _oscillation_trials(srate, frames, [0.0, 0.2, 0.5]) + trend * np.asarray([1.0, 0.7, 1.3])
    ch1 = _oscillation_trials(srate, frames, [0.1, 0.3, 0.4]) + trend * np.asarray([0.8, 1.1, 0.9])
    multi = np.stack([ch0, ch1], axis=0)
    common = dict(cycles=[3, 0.5], tlimits=tlimits, freqs=[8, 18], ntimesout=8)
    multi_off = timefreq(multi, srate, detrend="off", **common)
    multi_on = timefreq(multi, srate, detrend="on", **common)
    np.testing.assert_allclose(multi_on.tfdata, multi_off.tfdata, rtol=1e-12, atol=1e-12)

    single_off = timefreq(ch0, srate, frames=frames, detrend="off", **common)
    single_on = timefreq(ch0, srate, frames=frames, detrend="on", **common)
    assert not np.allclose(single_on.tfdata, single_off.tfdata)


def test_replace_zero_bins_matches_eeglab_guard():
    # timefreq.m 543-548: exact-zero bins take the smallest non-zero estimate.
    clean = np.asarray([[1 + 1j, 2 - 1j], [0.5 + 0.5j, 3 + 0j]])
    np.testing.assert_array_equal(_replace_zero_bins(clean), clean)
    with_zero = np.asarray([[0 + 0j, 2 - 1j], [0.5 + 0.5j, 3 + 0j]])
    filled = _replace_zero_bins(with_zero)
    assert filled[0, 0] == 0.5 + 0.5j
    np.testing.assert_array_equal(filled[0, 1:], with_zero[0, 1:])
    np.testing.assert_array_equal(filled[1, :], with_zero[1, :])


def test_timewarp_matches_eeglab_linear_interpolation_matrix():
    matrix = timewarp([1, 3, 5], [1, 2, 5])

    expected = np.asarray(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1 / 3, 2 / 3, 0],
            [0, 0, 0, 2 / 3, 1 / 3],
            [0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=1e-12)


@eeglab_test("unittesting_sigprocfunc/angtimewarp/sigprocfunc_angtimewarp_wrapperTest.m", "test_pass_5point_sinus")
@eeglab_test("unittesting_sigprocfunc/angtimewarp/sigprocfunc_angtimewarp_wrapperTest.m", "test_pass_diff_start")
@eeglab_test("unittesting_sigprocfunc/angtimewarp/sigprocfunc_angtimewarp_wrapperTest.m", "test_pass_spike")
def test_reference_angtimewarp(eeglab_backend):
    data = np.array([[0, 1, 0, -1, 0]], dtype=float)
    result = eeglab_backend("angtimewarp", np.array([[1, 3, 5]], dtype=float), np.array([[1, 5, 5]], dtype=float), data)
    assert_matlab_near(result, [[0, 0.5, 1, 0.5, 0]])
    # The source's remaining two numerical assertions are commented out.
    eeglab_backend("angtimewarp", np.array([[2, 3, 4]], dtype=float), np.array([[1, 3, 5]], dtype=float), data)
    eeglab_backend(
        "angtimewarp",
        np.array([[1, 3, 5]], dtype=float),
        np.array([[1, 1, 5]], dtype=float),
        np.array([[0, 5, 1000, -1000, 0]], dtype=float),
    )


def test_angtimewarp_interpolates_and_wraps_like_eeglab():
    angles = np.asarray([0, np.pi / 2, np.pi, -np.pi / 2, 0], dtype=float)

    warped = angtimewarp([1, 3, 5], [1, 2, 5], angles)

    np.testing.assert_allclose(warped, [0, np.pi, 0, -np.pi / 3, 0], rtol=1e-12, atol=1e-12)


def test_python_regression_angtimewarp_five_point_compression():
    warped = angtimewarp([1, 3, 5], [1, 5, 5], [0, 1, 0, -1, 0])

    np.testing.assert_allclose(warped, [0, 0.5, 1, 0.5, 0])


def test_python_regression_angtimewarp_implicit_synchronized_start():
    warped = angtimewarp([2, 3, 4], [1, 3, 5], [0, 1, 0, -1, 0])

    np.testing.assert_allclose(warped, [0, 0.5, 0, -0.5, -1])


def test_python_regression_angtimewarp_repeated_marker_wraps_large_angles():
    warped = angtimewarp([1, 3, 5], [1, 1, 5], [0, 5, 1000, -1000, 0])
    unwrapped = np.asarray([0, 0, -1000, -500, 0], dtype=float)
    expected = np.mod(unwrapped, 2 * np.pi)
    expected[expected > np.pi] -= 2 * np.pi

    np.testing.assert_allclose(warped, expected)


def test_tf_cycle_calc_converts_width_units_and_dialog_inventory():
    result = tf_cycle_calc(freqs=[10, 20], width=0.2, width_unit="fwhm_t")
    sigma2fwhm = 2 * np.sqrt(2 * np.log(2))
    expected_cycles = np.asarray([10, 20], dtype=float) * 2 * np.pi * 0.2 / sigma2fwhm

    np.testing.assert_allclose(result.cycles, expected_cycles, rtol=1e-12, atol=1e-12)
    assert result.widths_table.shape == (2, 8)
    assert result.columns == (
        "freq",
        "cycles",
        "fwhm_f",
        "fwhm_t",
        "2_sigma_f",
        "2_sigma_t",
        "sigma_f",
        "sigma_t",
    )

    cycle_result = tf_cycle_calc(freqs=[8, 12, 16], width=[3, 6], width_unit="cycles", log_spaced=False)
    np.testing.assert_allclose(cycle_result.cycles, [3, 4.5, 6], rtol=1e-12, atol=1e-12)

    spec = tf_cycle_calc_dialog_spec(freqs=[8, 16], width=[0.2, 0.3])
    controls = controls_by_tag(spec)
    assert spec.title == "Wavelet cycles calculator -- tf_cycle_calc()"
    assert controls["widthpop"].value == 1
    assert controls["freqedit"].value == "8 16"
    assert controls["widthedit"].value == "0.2 0.3"
    assert controls["plot"].callback.name == "tf_cycle_calc_plot"


def test_newcrossf_identical_synthetic_signals_have_unit_phase_coherence():
    srate = 128
    times = np.arange(0, 1, 1 / srate)
    trials = np.stack([np.sin(2 * np.pi * 12 * times), np.sin(2 * np.pi * 12 * times + 0.3)], axis=1)

    result = newcrossf(trials, trials, trials.shape[0], [0, 1000], srate, 0, freqs=[8, 16], plot="off")

    assert result.coherence.shape == result.phase.shape
    assert np.nanmean(result.coherence) > 0.99
    assert np.nanmax(np.abs(result.phase)) < 1e-10


def test_newcrossf_single_trial_switches_to_cross_spectrum():
    rng = np.random.default_rng(0)
    first = rng.normal(size=512)
    second = rng.normal(size=512)

    result = newcrossf(first, second, 512, [0, 511], 256, 0, plot="off")
    multi_trial = newcrossf(
        np.column_stack([first, rng.normal(size=512)]),
        np.column_stack([second, rng.normal(size=512)]),
        512,
        [0, 511],
        256,
        0,
        plot="off",
    )

    assert result.coherence.shape == result.phase.shape
    assert result.allcoher.shape[2] == 1
    assert np.nanmean(result.coherence) > 1.0
    assert np.nanmax(multi_trial.coherence) <= 1.0 + 1e-12


def test_pop_newtimef_channel_and_component_paths_are_replayable(sample_epoch, ica_epoch):
    result, command = pop_newtimef(sample_epoch, 1, 1, [-100, 200], [0], plot="off", return_com=True)
    component_result, component_command = pop_newtimef(ica_epoch, 0, 1, [-100, 200], [0], plot="off", return_com=True)

    assert result.ersp.ndim == 2
    assert component_result.ersp.ndim == 2
    assert "pop_newtimef(EEG, 1, 1" in command
    assert "pop_newtimef(EEG, 0, 1" in component_command
    _assert_python_command(command)
    _assert_python_command(component_command)
    namespace = {"EEG": sample_epoch, "pop_newtimef": pop_newtimef}
    replayed = eval(command, namespace)
    assert replayed.ersp.shape == result.ersp.shape


def test_pop_newtimef_timewarp_options_are_replayable(sample_epoch):
    trial_count = int(sample_epoch["trials"])
    first_marker = np.linspace(20, 40, trial_count)
    second_marker = np.linspace(70, 90, trial_count)
    markers = np.column_stack([first_marker, second_marker])

    result, command = pop_newtimef(
        sample_epoch,
        1,
        1,
        [-100, 200],
        [3, 0.8],
        freqs=[20, 30],
        nfreqs=2,
        timesout=10,
        timewarp=markers,
        timewarpms=[30, 80],
        timewarpidx=[1, 2],
        plot="off",
        return_com=True,
    )

    assert result.tfdata.shape == (result.freqs.size, result.times.size, trial_count)
    np.testing.assert_allclose(result.timewarp_markers, [31.25, 78.125], rtol=1e-12, atol=1e-12)
    assert "timewarp=" in command
    assert "timewarpms=[30, 80]" in command
    assert "timewarpidx=[1, 2]" in command
    _assert_python_command(command)
    namespace = {"EEG": sample_epoch, "pop_newtimef": pop_newtimef}
    replayed = eval(command, namespace)
    np.testing.assert_allclose(replayed.timewarp_markers, result.timewarp_markers, rtol=1e-12, atol=1e-12)


def test_timefreq_timestretch_ignores_duplicate_snaps_on_coarse_grid():
    srate = 128
    times = np.arange(128) / srate
    trials = np.stack(
        [
            np.sin(2 * np.pi * 10 * times),
            np.sin(2 * np.pi * 10 * times + 0.2),
        ],
        axis=1,
    )

    result = timefreq(
        trials,
        srate,
        frames=128,
        cycles=0,
        tlimits=[0, 1000],
        freqs=[5, 20],
        ntimesout=4,
        padratio=2,
        timestretch=(np.asarray([[1, 2], [1, 2]], dtype=float), np.asarray([1, 2], dtype=float)),
        verbose="off",
    )

    assert result.tfdata.shape[1] == result.times.size
    assert np.isfinite(np.abs(result.tfdata)).all()


def test_timefreq_frames_splits_single_channel_matrix_into_trials():
    srate = 128
    times = np.arange(128) / srate
    trials = np.stack(
        [
            np.sin(2 * np.pi * 10 * times),
            np.sin(2 * np.pi * 10 * times + 0.2),
        ],
        axis=1,
    )
    row_vector = trials.T.reshape(1, -1)

    row_result = timefreq(row_vector, srate, frames=128, cycles=0, freqs=[5, 20], ntimesout=8, verbose="off")
    column_result = timefreq(row_vector.T, srate, frames=128, cycles=0, freqs=[5, 20], ntimesout=8, verbose="off")
    matrix_result = timefreq(trials, srate, frames=128, cycles=0, freqs=[5, 20], ntimesout=8, verbose="off")

    assert row_result.tfdata.shape[-1] == column_result.tfdata.shape[-1] == matrix_result.tfdata.shape[-1] == 2
    np.testing.assert_allclose(row_result.tfdata, matrix_result.tfdata, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(column_result.tfdata, matrix_result.tfdata, rtol=1e-12, atol=1e-12)


def test_pop_newcrossf_channel_and_component_paths_are_replayable(sample_epoch, ica_epoch):
    result, command = pop_newcrossf(sample_epoch, 1, 1, 2, [-100, 200], [0], plot="off", return_com=True)
    component_result, component_command = pop_newcrossf(
        ica_epoch, 0, 1, 2, [-100, 200], [0], plot="off", return_com=True
    )

    assert result.coherence.ndim == 2
    assert component_result.coherence.ndim == 2
    assert "pop_newcrossf(EEG, 1, 1, 2" in command
    assert "pop_newcrossf(EEG, 0, 1, 2" in component_command
    _assert_python_command(command)
    _assert_python_command(component_command)


def test_pop_signalstat_sample_and_component_paths(sample_eeg, ica_epoch):
    result, command = pop_signalstat(sample_eeg, 1, 1, 5, return_com=True)
    component_result, component_command = pop_signalstat(ica_epoch, 0, 1, 5, return_com=True)

    assert np.isfinite(result.mean)
    assert np.isfinite(component_result.mean)
    assert result.trimmed_indices.size > 0
    assert "pop_signalstat(EEG, 1, 1, 5)" == command
    assert "pop_signalstat(EEG, 0, 1, 5)" == component_command
    _assert_python_command(command)
    plt.close(result.figure)
    plt.close(component_result.figure)


def test_pop_eventstat_extracts_sample_event_latencies(sample_eeg):
    values = event_values(sample_eeg, "latency", type=["square"])
    result, command = pop_eventstat(sample_eeg, "latency", ["square"], [], 5, return_com=True)

    assert values.size > 0
    assert result.mean == pytest.approx(float(np.mean(values)))
    assert "pop_eventstat(EEG, 'latency', ['square'], [], 5)" == command
    _assert_python_command(command)
    plt.close(result.figure)


def test_newtimef_significance_flags_event_related_effect():
    # A phase-locked post-stimulus burst must test significant against the
    # pre-stimulus baseline null; EEGLAB ranks each cell against a per-frequency
    # baseline distribution, so the effect survives while the baseline does not.
    srate = 256
    frames = 512
    tlimits = [-1000, 1000]
    times = np.linspace(tlimits[0], tlimits[1], frames) / 1000.0
    rng = np.random.default_rng(0)
    burst = np.exp(-((times - 0.30) ** 2) / (2 * 0.12**2)) * (times > 0)
    data = np.stack(
        [0.8 * rng.standard_normal(frames) + 1.6 * np.sin(2 * np.pi * 10 * times) * burst for _ in range(40)],
        axis=1,
    )

    result = newtimef(
        data,
        frames,
        tlimits,
        srate,
        [3, 0.5],
        freqs=[5, 45],
        nfreqs=50,
        baseline=[-1000, 0],
        alpha=0.05,
        rng=0,
        plot="off",
    )

    row = int(np.argmin(np.abs(result.freqs - 10)))
    post = result.times > 150
    pre = result.times < -150
    assert result.itc_significant[row, post].mean() > 0.7  # the 10 Hz post-stimulus band is significant
    assert result.itc_significant[row, pre].mean() < 0.1  # the pre-stimulus baseline is not
    # a sane overall fraction -- not the near-zero of the pre-fix full-data-null bug, nor everything
    assert 0.02 < result.itc_significant.mean() < 0.35
    assert 0.02 < result.ersp_significant.mean() < 0.35
    # EEGLAB's compute_pvals is two-sided for ITC, so the effect-free baseline flags ~alpha of cells
    # (both tails) as chance false positives -- roughly double the ~alpha/2 an upper-tail test gives.
    baseline = result.times < 0
    assert 0.03 < result.itc_significant[:, baseline].mean() < 0.08


def test_legacy_timef_and_crossf_wrappers_return_replayable_history(sample_epoch):
    timef_result, timef_command = pop_timef(sample_epoch, 1, 1, [-100, 200], [0], plot="off", return_com=True)
    crossf_result, crossf_command = pop_crossf(sample_epoch, 1, 1, 2, [-100, 200], [0], plot="off", return_com=True)

    assert timef_result.ersp.ndim == 2
    assert crossf_result.coherence.ndim == 2
    assert timef_command.startswith("pop_timef(EEG, 1, 1")
    assert crossf_command.startswith("pop_crossf(EEG, 1, 1, 2")
    _assert_python_command(timef_command)
    _assert_python_command(crossf_command)


def test_bootstrap_threshold_matches_eeglab_tail_mean_formula():
    # EEGLAB bootstat thresholds the sorted null at i = round(naccu*alpha) and averages the
    # i most extreme surrogates on each requested side (bootstat.m accarray1/accarray2). Pin
    # that tail selection and averaging with hand-computed values so a switch to a
    # percentile-interpolation or floor(i) rule would fail loudly.
    surrogates = np.arange(1.0, 21.0).reshape(20, 1)  # 20 sorted surrogates, one frequency

    # alpha=0.1 -> i = round(20 * 0.1) = 2: mean of the two most extreme surrogates per side.
    assert float(bootstrap_threshold(surrogates, alpha=0.1, bootside="upper")) == pytest.approx(19.5)  # mean(19, 20)
    np.testing.assert_allclose(bootstrap_threshold(surrogates, alpha=0.1, bootside="both"), [1.5, 19.5])
    # alpha=0.05 -> i = round(20 * 0.05) = 1: the single most extreme surrogate per side.
    np.testing.assert_allclose(bootstrap_threshold(surrogates, alpha=0.05, bootside="both"), [1.0, 20.0])
    # Complex surrogates are thresholded on magnitude (EEGLAB accarray = sqrt(x .* conj(x))).
    complex_col = (np.arange(1.0, 21.0) * np.exp(1j * np.arange(20))).reshape(20, 1)
    assert float(bootstrap_threshold(complex_col, alpha=0.05, bootside="upper")) == pytest.approx(20.0)


def test_timefreq_shared_bootstrap_helpers_cover_newtimef_and_newcrossf_paths():
    times = np.asarray([-100.0, 0.0, 0.5, 100.0, 200.0])
    baseln = np.asarray([0, 1], dtype=int)

    np.testing.assert_array_equal(bootstrap_indices(times, baseline=0, baseboot=[], baseln=baseln), baseln)
    np.testing.assert_array_equal(bootstrap_indices(times, baseline=np.nan, baseboot=1, baseln=baseln), [0, 1])
    np.testing.assert_array_equal(bootstrap_indices(times, baseline=np.nan, baseboot=0, baseln=baseln), [])
    np.testing.assert_array_equal(bootstrap_indices(times, baseline=np.nan, baseboot=[50, 200], baseln=baseln), [3, 4])
    np.testing.assert_array_equal(bootstrap_indices(times, baseboot=1, baseln=None, limit_to_baseboot=True), [0, 1, 2])

    surrogates = np.arange(24, dtype=float).reshape(2, 3, 4)
    np.testing.assert_allclose(
        thresholds_by_frequency(surrogates, alpha=0.1, bootside="both"),
        _thresholds_by_frequency(surrogates, alpha=0.1, both=True),
    )
    np.testing.assert_allclose(
        thresholds_by_frequency(surrogates, alpha=0.1, bootside="upper"),
        _upper_thresholds_by_frequency(surrogates, alpha=0.1),
    )
    assert threshold_vector(2.0, (3, 4)).shape == (3, 4)
    assert threshold_vector(np.asarray([1.0, 2.0, 3.0]), (3, 4)).shape == (3, 1)

    values = np.arange(24, dtype=float).reshape(2, 3, 4)
    shuffled = resample_trials(values, np.random.default_rng(0), "shuffle")
    randomized = resample_trials(values.astype(complex), np.random.default_rng(0), "rand", complex_phase=True)
    assert shuffled.shape == values.shape
    np.testing.assert_allclose(np.abs(randomized), np.abs(values))


def test_bootstat_basevect_uses_eeglab_one_based_indices():
    data = np.arange(12, dtype=float).reshape(2, 3, 2)
    seen = []

    def statistic(value):
        seen.append(value.copy())
        return value[:, 0, :]

    bootstat(data, statistic=statistic, basevect=[1], shuffledim=[2], naccu=1, rng=0)

    np.testing.assert_array_equal(seen[0], data[:, :1, :])
    with pytest.raises(ValueError, match="1-based"):
        bootstat(data, statistic=statistic, basevect=[0], naccu=1, rng=0)


def test_empirical_pvalue_conventions_are_intentionally_distinct():
    distribution = np.asarray([1.0, 2.0, 3.0, 4.0])
    observed = 5.0

    assert pac_empirical_pvalue(distribution, observed) == pytest.approx(1 / 5)
    np.testing.assert_allclose(
        stat_surrogate_pvals(distribution[np.newaxis, :], np.asarray([observed]), "right"), [0.0]
    )
    np.testing.assert_allclose(exact_p_values(observed, distribution, center=0.0), 0.0)


def test_ramberg_schmeiser_helpers_cover_analytic_cases():
    uniform_lambdas = np.asarray([0.0, 2.0, 1.0, 1.0])

    assert rspfunc(0.75, uniform_lambdas, 0.25) == pytest.approx(0.0, abs=1e-12)
    assert rsget(uniform_lambdas, 0.25) == pytest.approx(0.75, abs=1e-8)
    assert rspdfsolv([1.0, 1.0], 0.0, 1.8) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(rsadjust(1.0, 1.0, 0.0, 1.0 / 12.0, 0.0), uniform_lambdas)
    np.testing.assert_allclose(
        rsadjust(-0.1, 1.45, 0.25, 0.5, 1.0),
        [-2.1913486194442604, 0.28793423446627836, -0.1, 1.45],
        rtol=1e-12,
        atol=1e-12,
    )

    pvalue, cumulants, lambdas, _chi2 = rsfit(np.linspace(-1.0, 1.0, 101), 0.0, return_details=True)
    assert pvalue == pytest.approx(0.5, abs=1e-8)
    np.testing.assert_allclose(cumulants[:3], [0.0, 0.34, 0.0], atol=1e-12)
    np.testing.assert_allclose(lambdas[[0, 2, 3]], [0.0, 1.00098197, 1.00098197], atol=1e-6)


def test_correctfit_applies_gamma_parameters_and_zero_mode():
    corrected, shape, scale, zero_frequency = correctfit(0.01, gamparams=[2.0, 0.5, 0.25])

    expected = 1.0 - stats.gamma.cdf(-np.log10(0.01) + 1.0e-10, 2.0, scale=0.5)
    assert corrected == pytest.approx(expected)
    assert (shape, scale, zero_frequency) == pytest.approx((2.0, 0.5, 0.25))
    assert correctfit(0.0, gamparams=[2.0, 0.5, 0.25])[0] == pytest.approx(0.25)
    assert correctfit(0.0, gamparams=[2.0, 0.5, 0.25], zeromode="off")[0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("action", "module_path", "expected_kwargs", "command"),
    [
        (
            "pop_newtimef:channels",
            "eegprep.functions.popfunc.pop_newtimef.pop_newtimef",
            {"typeproc": 1, "return_com": True},
            "pop_newtimef(EEG, 1, 1)",
        ),
        (
            "pop_newcrossf:components",
            "eegprep.functions.popfunc.pop_newcrossf.pop_newcrossf",
            {"typeproc": 0, "return_com": True},
            "pop_newcrossf(EEG, 0, 1, 2)",
        ),
    ],
)
def test_timefreq_statistics_menu_dispatch_records_history(sample_eeg, action, module_path, expected_kwargs, command):
    session = EEGPrepSession()
    session.store_current(sample_eeg, new=True)
    stored_eeg = session.EEG
    dispatcher = MenuActionDispatcher(session)

    with mock.patch(module_path, return_value=("figure", command)) as pop_function:
        dispatcher.dispatch(action)

    pop_function.assert_called_once()
    assert len(pop_function.call_args.args) == 1
    assert pop_function.call_args.args[0] is stored_eeg
    assert pop_function.call_args.kwargs == expected_kwargs
    assert session.EEG is stored_eeg
    assert session.ALLCOM[-1] == command


def test_signalstat_matches_numpy_for_known_vector():
    values = np.asarray([1, 2, 3, 4, 100], dtype=float)
    result = signalstat(values, plotlab=0, percent=40)

    assert result.mean == pytest.approx(np.mean(values))
    assert result.std == pytest.approx(np.std(values, ddof=1))
    assert result.median == pytest.approx(np.median(values))
    assert result.zlow == pytest.approx(1.5)
    assert result.zhigh == pytest.approx(52.0)
    assert result.trimmed_indices.tolist() == [1, 2, 3]


@pytest.mark.matlab
def test_signalstat_statistics_match_eeglab(tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB reference checkout not available")

    values = np.asarray([1.0, 2.5, -3.0, 4.25, 5.5, 9.0, 12.0, 20.0])
    output = tmp_path / "signalstat.mat"
    engine = matlab_engine.start_matlab()
    try:
        engine.addpath(str(eeglab_root / "functions" / "sigprocfunc"), nargout=0)
        engine.eval(
            f"""
            data = [{_matlab_vector(values)}];
            [M,SD,sk,k,med,zlow,zhi,tM,tSD,tndx,ksh] = signalstat(data, 0, [], 10);
            save('{_matlab_string(output)}', 'M', 'SD', 'sk', 'k', 'med', 'zlow', 'zhi', 'tM', 'tSD', 'tndx', 'ksh');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    result = signalstat(values, plotlab=0, percent=10)
    matlab = scipy.io.loadmat(output, squeeze_me=True)
    np.testing.assert_allclose(result.mean, matlab["M"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.std, matlab["SD"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.skewness, matlab["sk"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.kurtosis, matlab["k"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.median, matlab["med"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.zlow, matlab["zlow"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.zhigh, matlab["zhi"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.trimmed_mean, matlab["tM"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.trimmed_std, matlab["tSD"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result.trimmed_indices + 1, np.asarray(matlab["tndx"], dtype=int).ravel())


@pytest.mark.matlab
def test_timefreq_helpers_match_eeglab_deterministic_outputs(tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB reference checkout not available")

    srate = 128.0
    sample_times = np.arange(128) / srate
    trials = np.stack(
        [
            np.sin(2 * np.pi * 10 * sample_times),
            np.sin(2 * np.pi * 10 * sample_times + 0.2),
        ],
        axis=1,
    )
    power = np.arange(1, 1 + 3 * 5 * 4, dtype=float).reshape(3, 5, 4) / 10.0
    time_values = np.asarray([-200, -100, 0, 100, 200], dtype=float)
    inputs = tmp_path / "timefreq_inputs.mat"
    output = tmp_path / "timefreq_outputs.mat"
    scipy.io.savemat(inputs, {"data": trials, "P": power, "time_values": time_values})

    engine = matlab_engine.start_matlab()
    try:
        engine.addpath(engine.genpath(str(eeglab_root / "functions")), nargout=0)
        engine.eval(
            f"""
            load('{_matlab_string(inputs)}');
            wavelet2 = dftfilt2([6 10], [3 5], {srate}, 'linear', 'morlet');
            dft2wav1 = wavelet2{{1}};
            dft2wav2 = wavelet2{{2}};
            [wavelet,cycles,freqresol,timeresol] = dftfilt3([6 10], [3 5], {srate}, 'cycleinc', 'linear');
            wav1 = wavelet{{1}};
            wav2 = wavelet{{2}};
            [tf,freqs,times] = timefreq(data, {srate}, 'cycles', 0, 'tlimits', [0 1000], ...
                'freqs', [5 20], 'ntimesout', 12, 'padratio', 2, 'verbose', 'off');
            [tfstretch,stretchfreqs,stretchtimes] = timefreq(data, {srate}, 'cycles', 0, 'tlimits', [0 1000], ...
                'freqs', [5 20], 'ntimesout', 12, 'padratio', 2, ...
                'timestretch', {{[20 80; 24 76], [22; 78]}}, 'verbose', 'off');
            [PP,baseln,mbase] = newtimefbaseln(P, time_values, 'baseline', [-200 0], ...
                'basenorm', 'off', 'trialbase', 'off', 'verbose', 'off');
            tw = timewarp([1 3 5], [1 2 5]);
            aw = angtimewarp([1 3 5], [1 2 5], [0 pi/2 pi -pi/2 0]);
            [calc_cycles, widths_table] = tf_cycle_calc('freqs', [10 20], 'width', 0.2, 'width_unit', 'fwhm_t');
            save('{_matlab_string(output)}', 'wav1', 'wav2', 'cycles', 'freqresol', 'timeresol', ...
                'dft2wav1', 'dft2wav2', 'tf', 'freqs', 'times', 'tfstretch', 'stretchfreqs', 'stretchtimes', ...
                'PP', 'baseln', 'mbase', ...
                'tw', 'aw', 'calc_cycles', 'widths_table');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    matlab = scipy.io.loadmat(output, squeeze_me=True)
    dft2_wavelets = dftfilt2([6, 10], [3, 5], srate)
    wavelets, py_cycles, py_freqresol, py_timeresol = dftfilt3([6, 10], [3, 5], srate, cycleinc="linear")
    decomposition = timefreq(
        trials,
        srate,
        frames=128,
        cycles=0,
        tlimits=[0, 1000],
        freqs=[5, 20],
        ntimesout=12,
        padratio=2,
        verbose="off",
    )
    stretch_decomposition = timefreq(
        trials,
        srate,
        frames=128,
        cycles=0,
        tlimits=[0, 1000],
        freqs=[5, 20],
        ntimesout=12,
        padratio=2,
        timestretch=(np.asarray([[20, 80], [24, 76]], dtype=float), np.asarray([22, 78], dtype=float)),
        verbose="off",
    )
    py_power, py_baseln, py_mbase = newtimefbaseln(
        power,
        time_values,
        baseline=[-200, 0],
        basenorm="off",
        trialbase="off",
    )

    np.testing.assert_allclose(dft2_wavelets[0], matlab["dft2wav1"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dft2_wavelets[1], matlab["dft2wav2"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wavelets[0], matlab["wav1"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wavelets[1], matlab["wav2"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(py_cycles, matlab["cycles"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(py_freqresol, matlab["freqresol"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(py_timeresol, matlab["timeresol"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(decomposition.freqs, np.asarray(matlab["freqs"]).ravel(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(decomposition.times, np.asarray(matlab["times"]).ravel(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(decomposition.tfdata, matlab["tf"], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        stretch_decomposition.freqs, np.asarray(matlab["stretchfreqs"]).ravel(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        stretch_decomposition.times, np.asarray(matlab["stretchtimes"]).ravel(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(stretch_decomposition.tfdata, matlab["tfstretch"], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(py_power, matlab["PP"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(py_baseln + 1, np.asarray(matlab["baseln"], dtype=int).ravel())
    np.testing.assert_allclose(py_mbase, matlab["mbase"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(timewarp([1, 3, 5], [1, 2, 5]), matlab["tw"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        angtimewarp([1, 3, 5], [1, 2, 5], [0, np.pi / 2, np.pi, -np.pi / 2, 0]),
        np.asarray(matlab["aw"]).ravel(),
        rtol=1e-12,
        atol=1e-12,
    )
    cycle_result = tf_cycle_calc(freqs=[10, 20], width=0.2, width_unit="fwhm_t")
    np.testing.assert_allclose(cycle_result.cycles, np.asarray(matlab["calc_cycles"]).ravel(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cycle_result.widths_table, matlab["widths_table"], rtol=1e-12, atol=1e-12)


@pytest.mark.matlab
def test_newtimef_matches_eeglab_ersp_itc_and_pvalues(tmp_path):
    # End-to-end parity for the assembled newtimef outputs (the low-level helpers are covered
    # above): the ERSP (dB) and complex ITC are deterministic, so they must match EEGLAB tightly.
    # The two-sided baseline p-value (the significance mask's core) is checked against EEGLAB's
    # compute_pvals formula, reimplemented inline below because it is a private subfunction of
    # newtimef.m and cannot be called directly, on fixed observed/null arrays -- no bootstrap
    # randomness on either side.
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB reference checkout not available")

    srate = 128.0
    n_frames = 128
    tlimits = [-500, 500]  # 1 s epoch; the 5 Hz wavelet eats ~300 ms at each edge
    sample_times = np.arange(n_frames) / srate
    envelope = 1.0 + (sample_times > 0.5)  # amplitude step mid-epoch -> a non-trivial ERSP
    trials = np.stack(
        [
            envelope * np.sin(2 * np.pi * 10 * sample_times + phase) + 0.5 * np.sin(2 * np.pi * 6 * sample_times)
            for phase in np.linspace(0.0, 1.2, 12)
        ],
        axis=1,
    )
    # Request output times on exact frame centers well inside the valid decomposition range, so
    # both engines select identical frames (the scalar-timesout auto-grid can round one interior
    # point to a neighbouring frame, MATLAB round() vs NumPy half-to-even -- not what we test here).
    frame_times = tlimits[0] + np.arange(n_frames) * (tlimits[1] - tlimits[0]) / (n_frames - 1)
    timesout = frame_times[[44, 52, 60, 68, 76, 82]]
    # Fixed arrays for the p-value comparison; both sides consume the identical saved values.
    generator = np.random.default_rng(0)
    observed = generator.standard_normal((6, 10))
    null = generator.standard_normal((6, 50))

    inputs = tmp_path / "newtimef_inputs.mat"
    output = tmp_path / "newtimef_outputs.mat"
    scipy.io.savemat(inputs, {"data": trials, "obs": observed, "null": null, "timesout": timesout})

    engine = matlab_engine.start_matlab()
    try:
        engine.addpath(engine.genpath(str(eeglab_root / "functions")), nargout=0)
        engine.eval(
            f"""
            load('{_matlab_string(inputs)}');
            figure('visible','off');
            [P,R,mbase,times,freqs] = newtimef(data, {n_frames}, [{tlimits[0]} {tlimits[1]}], {srate}, [3 0.5], ...
                'freqs', [5 20], 'nfreqs', 8, 'timesout', timesout, 'baseline', [-200 0], ...
                'plotphase', 'off', 'verbose', 'off');
            surrog = repmat(reshape(null, [size(null,1) 1 size(null,2)]), [1 size(obs,2) 1]);
            surrog = sort(surrog, 3);
            surrog(:,:,end+1) = obs;
            [~, idx] = sort(surrog, 3);
            [~, mx] = max(idx, [], 3);
            pupper = 1 - (mx - 0.5) / size(surrog, 3);
            pvals = 2 * min(pupper, 1 - pupper);
            save('{_matlab_string(output)}', 'P', 'R', 'mbase', 'times', 'freqs', 'pvals');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    result = newtimef(
        trials,
        n_frames,
        tlimits,
        srate,
        [3, 0.5],
        freqs=[5, 20],
        nfreqs=8,
        timesout=timesout,
        baseline=[-200, 0],
        plotphase="off",
        plot="off",
    )
    py_pvals = _baseline_pvalues(observed, null.T[:, :, None])

    matlab = scipy.io.loadmat(output, squeeze_me=True)
    np.testing.assert_allclose(result.freqs, np.asarray(matlab["freqs"]).ravel(), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(result.times, np.asarray(matlab["times"]).ravel(), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(result.ersp, matlab["P"], rtol=1e-6, atol=1e-6)  # ERSP (dB)
    np.testing.assert_allclose(result.itc, matlab["R"], rtol=1e-6, atol=1e-6)  # complex ITC
    np.testing.assert_allclose(py_pvals, matlab["pvals"], rtol=1e-12, atol=1e-12)  # two-sided compute_pvals
    np.testing.assert_allclose(  # baseline spectrum in dB (EEGLAB mbase) -- the default log/baseline case
        np.asarray(result.powbase).ravel(), np.asarray(matlab["mbase"]).ravel(), rtol=1e-6, atol=1e-6
    )


@pytest.mark.matlab
def test_newtimef_scale_and_baseline_modes_match_eeglab(tmp_path):
    # Parity for the option paths the deterministic end-to-end test above does not exercise:
    # absolute power scale, baseline normalization (basenorm), single-trial baseline (trialbase 'full'),
    # and the cycles=0 short-time FFT path. Each case is deterministic, so P (ERSP), R (complex ITC),
    # and mbase (baseline spectrum) match EEGLAB tightly. mbase in particular guards the units EEGLAB
    # returns: dB for the default log scale (newtimef.m:1399), absolute power for abs/basenorm/trialbase.
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB reference checkout not available")

    srate = 128.0
    n_frames = 128
    tlimits = [-500, 500]
    sample_times = np.arange(n_frames) / srate
    envelope = 1.0 + (sample_times > 0.5)  # amplitude step mid-epoch -> a non-trivial ERSP
    trials = np.stack(
        [
            envelope * np.sin(2 * np.pi * 10 * sample_times + phase) + 0.5 * np.sin(2 * np.pi * 6 * sample_times)
            for phase in np.linspace(0.0, 1.2, 12)
        ],
        axis=1,
    )
    # Output times on exact frame centers well inside the valid range, so both engines pick identical frames.
    frame_times = tlimits[0] + np.arange(n_frames) * (tlimits[1] - tlimits[0]) / (n_frames - 1)
    timesout = frame_times[[44, 52, 60, 68, 76, 82]]

    inputs = tmp_path / "newtimef_modes_inputs.mat"
    output = tmp_path / "newtimef_modes_outputs.mat"
    scipy.io.savemat(inputs, {"data": trials, "timesout": timesout})

    engine = matlab_engine.start_matlab()
    try:
        engine.addpath(engine.genpath(str(eeglab_root / "functions")), nargout=0)
        engine.eval(
            f"""
            load('{_matlab_string(inputs)}');
            set(0, 'DefaultFigureVisible', 'off');
            common = {{'freqs', [5 20], 'nfreqs', 8, 'timesout', timesout, 'baseline', [-200 0], ...
                      'plotphase', 'off', 'verbose', 'off'}};
            [abs_P, abs_R, abs_mbase, abs_times, abs_freqs] = ...
                newtimef(data, {n_frames}, [{tlimits[0]} {tlimits[1]}], {srate}, [3 0.5], 'scale', 'abs', common{{:}});
            [bn_P, bn_R, bn_mbase] = ...
                newtimef(data, {n_frames}, [{tlimits[0]} {tlimits[1]}], {srate}, [3 0.5], 'basenorm', 'on', common{{:}});
            [tb_P, tb_R, tb_mbase] = ...
                newtimef(data, {n_frames}, [{tlimits[0]} {tlimits[1]}], {srate}, [3 0.5], 'trialbase', 'full', common{{:}});
            [fft_P, fft_R, fft_mbase] = ...
                newtimef(data, {n_frames}, [{tlimits[0]} {tlimits[1]}], {srate}, 0, 'padratio', 2, common{{:}});
            close all;
            save('{_matlab_string(output)}', 'abs_P', 'abs_R', 'abs_mbase', 'abs_times', 'abs_freqs', ...
                 'bn_P', 'bn_R', 'bn_mbase', 'tb_P', 'tb_R', 'tb_mbase', 'fft_P', 'fft_R', 'fft_mbase');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    common = dict(freqs=[5, 20], nfreqs=8, timesout=timesout, baseline=[-200, 0], plotphase="off", plot="off")
    results = {
        "abs": newtimef(trials, n_frames, tlimits, srate, [3, 0.5], scale="abs", **common),
        "bn": newtimef(trials, n_frames, tlimits, srate, [3, 0.5], basenorm="on", **common),
        "tb": newtimef(trials, n_frames, tlimits, srate, [3, 0.5], trialbase="full", **common),
        "fft": newtimef(trials, n_frames, tlimits, srate, 0, padratio=2, **common),
    }

    matlab = scipy.io.loadmat(output, squeeze_me=True)
    np.testing.assert_allclose(results["abs"].freqs, np.asarray(matlab["abs_freqs"]).ravel(), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(results["abs"].times, np.asarray(matlab["abs_times"]).ravel(), rtol=1e-9, atol=1e-9)
    for case in ("abs", "bn", "tb", "fft"):
        np.testing.assert_allclose(results[case].ersp, matlab[f"{case}_P"], rtol=1e-6, atol=1e-6)  # ERSP
        np.testing.assert_allclose(results[case].itc, matlab[f"{case}_R"], rtol=1e-6, atol=1e-6)  # complex ITC
        np.testing.assert_allclose(  # baseline spectrum: dB for the log FFT case, absolute power otherwise
            np.asarray(results[case].powbase, dtype=float).ravel(),
            np.asarray(matlab[f"{case}_mbase"], dtype=float).ravel(),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.matlab
def test_timefreq_negative_ntimesout_times_match_eeglab(tmp_path):
    # Ground-truth the negative-ntimesout subsample grid against real EEGLAB timefreq (not a
    # hand-encoded colon formula): the trickiest off-by-one in the decomposition is np.arange's
    # exclusive stop vs MATLAB's inclusive a:step:b. frames/winsize/nsub are chosen so EEGLAB's
    # colon endpoint (length-ceil(winsize/2)-1) lands exactly on a grid point, so dropping or
    # adding the trailing time would change the output length and fail here.
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB reference checkout not available")

    srate = 128.0
    frames = 122  # -> EEGLAB colon stop 122-ceil(16/2)-1 = 113 sits on the 13:10:113 grid
    tlimits = [0.0, 1000.0]
    winsize = 16
    nsub = 10
    trials = _oscillation_trials(srate, frames, [0.0, 0.2, 0.5])

    inputs = tmp_path / "timefreq_subsample_inputs.mat"
    output = tmp_path / "timefreq_subsample_outputs.mat"
    scipy.io.savemat(inputs, {"data": trials})

    engine = matlab_engine.start_matlab()
    try:
        engine.addpath(engine.genpath(str(eeglab_root / "functions")), nargout=0)
        engine.eval(
            f"""
            load('{_matlab_string(inputs)}');
            [~, freqs, times] = timefreq(data, {srate}, 'cycles', 0, 'tlimits', [{tlimits[0]} {tlimits[1]}], ...
                'winsize', {winsize}, 'ntimesout', {-nsub}, 'freqs', [5 20], 'padratio', 2, ...
                'detrend', 'off', 'causal', 'off', 'verbose', 'off');
            save('{_matlab_string(output)}', 'times', 'freqs');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    result = timefreq(
        trials,
        srate,
        frames=frames,
        cycles=0,
        tlimits=tlimits,
        freqs=[5, 20],
        winsize=winsize,
        padratio=2,
        ntimesout=-nsub,
    )

    matlab = scipy.io.loadmat(output, squeeze_me=True)
    np.testing.assert_allclose(result.times, np.asarray(matlab["times"]).ravel(), rtol=1e-9, atol=1e-9)


def _trial_common_baseline_power(rng, n_freq, n_base, n_trials, noise):
    # Positive baseline power whose time course is largely shared across trials, so an
    # average-then-resample null (spread ~ std of the trial-mean) diverges from EEGLAB's
    # permute-then-average null (the shared structure averages out across trials).
    shared = rng.gamma(3.0, 1.0, size=(n_freq, n_base))[:, :, None]
    return shared + noise * rng.gamma(3.0, 1.0, size=(n_freq, n_base, n_trials))


def test_bootstrap_power_null_is_not_degenerate():
    # The ERSP null averages a per-trial permutation of the baseline time course, so each
    # exemplar is a fresh trial-mean -- unlike the old with-replacement resample of the fixed
    # trial-mean spectrum, which produced at most n_base distinct null values per frequency.
    rng = np.random.default_rng(0)
    n_freq, n_base, n_trials = 4, 20, 15
    power = _trial_common_baseline_power(rng, n_freq, n_base, n_trials, noise=0.01)
    _, baseline_null = _bootstrap_power(power, "abs", alpha=0.05, naccu=200, base_indices=np.arange(n_base), rng=0)
    for freq in range(n_freq):
        assert np.unique(baseline_null[:, freq, :]).size > n_base


@pytest.mark.matlab
def test_bootstrap_power_null_matches_eeglab_bootstat(tmp_path):
    # The ERSP significance null must match EEGLAB bootstat's 'shuffle' permutation in
    # distribution. Bootstrap is random, so compare the converged per-frequency null spread
    # on identical trial-common baseline power (where average-then-resample would diverge)
    # within a loose tolerance, against real EEGLAB bootstat.
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB reference checkout not available")

    rng = np.random.default_rng(0)
    n_freq, n_base, n_trials = 5, 24, 16
    naccu = 3000
    power = _trial_common_baseline_power(rng, n_freq, n_base, n_trials, noise=0.2)

    inputs = tmp_path / "bootstrap_power_inputs.mat"
    output = tmp_path / "bootstrap_power_outputs.mat"
    scipy.io.savemat(inputs, {"P": power})

    engine = matlab_engine.start_matlab()
    try:
        engine.addpath(engine.genpath(str(eeglab_root / "functions")), nargout=0)
        engine.eval(
            f"""
            load('{_matlab_string(inputs)}');
            [~, ~, Pboottrials] = bootstat(P, 'mean(arg1,3);', 'boottype', 'shuffle', ...
                'shuffledim', 2, 'basevect', 1:size(P,2), 'naccu', {naccu}, 'alpha', 0.05, ...
                'dimaccu', 2, 'bootside', 'both');
            null_std = std(Pboottrials, 0, 1);
            save('{_matlab_string(output)}', 'null_std');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    _, baseline_null = _bootstrap_power(power, "abs", alpha=0.05, naccu=naccu, base_indices=np.arange(n_base), rng=0)
    py_null_std = np.std(baseline_null, axis=(0, 2))

    matlab = scipy.io.loadmat(output, squeeze_me=True)
    matlab_null_std = np.asarray(matlab["null_std"], dtype=float).ravel()
    np.testing.assert_allclose(py_null_std, matlab_null_std, rtol=0.1)


@pytest.mark.matlab
def test_bootstrap_itc_null_matches_eeglab_bootstat(tmp_path):
    # The ITC significance null must match EEGLAB bootstat's 'shuffle' permutation in
    # distribution: shuffling each trial's baseline time course breaks the inter-trial phase
    # alignment, and ITC is recomputed. Bootstrap is random, so compare the converged
    # per-frequency null spread within a loose tolerance against real EEGLAB bootstat on
    # identical complex tf estimates (newtimef.m ITC path, phasecoher normalization).
    # Note: naccu here counts full shuffles, whereas EEGLAB's dimaccu accumulates over time bins
    # (~naccu/ntimes shuffles); the pooled distributions coincide, which is what this test checks.
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB reference checkout not available")

    rng = np.random.default_rng(0)
    n_freq, n_base, n_trials = 5, 24, 16
    naccu = 3000
    # Complex tf estimates with partial inter-trial phase coherence (shared phase per time
    # bin plus per-trial jitter), so the shuffle null has a non-trivial per-frequency spread.
    mag = rng.gamma(3.0, 1.0, size=(n_freq, n_base, n_trials))
    shared_phase = rng.uniform(-np.pi, np.pi, size=(n_freq, n_base))[:, :, None]
    noise_phase = 0.6 * rng.standard_normal((n_freq, n_base, n_trials))
    tf = mag * np.exp(1j * (shared_phase + noise_phase))

    inputs = tmp_path / "bootstrap_itc_inputs.mat"
    output = tmp_path / "bootstrap_itc_outputs.mat"
    scipy.io.savemat(inputs, {"tf": tf})

    engine = matlab_engine.start_matlab()
    try:
        engine.addpath(engine.genpath(str(eeglab_root / "functions")), nargout=0)
        engine.eval(
            f"""
            load('{_matlab_string(inputs)}');
            inputdata = tf ./ sqrt(tf .* conj(tf));  % phasecoher normalization (newtimef.m)
            [~, ~, Rboottrials] = bootstat(inputdata, 'mean(arg1,3);', 'boottype', 'shuffle', ...
                'basevect', 1:size(tf,2), 'naccu', {naccu}, 'alpha', 0.05, ...
                'dimaccu', 2, 'bootside', 'upper');
            null_std = std(Rboottrials, 0, 1);
            save('{_matlab_string(output)}', 'null_std');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    _, baseline_null = _bootstrap_itc(tf, "phasecoher", alpha=0.05, naccu=naccu, base_indices=np.arange(n_base), rng=0)
    py_null_std = np.std(baseline_null, axis=(0, 2))

    matlab = scipy.io.loadmat(output, squeeze_me=True)
    matlab_null_std = np.asarray(matlab["null_std"], dtype=float).ravel()
    np.testing.assert_allclose(py_null_std, matlab_null_std, rtol=0.1)


@pytest.mark.matlab
def test_ramberg_schmeiser_helpers_match_eeglab_deterministic_outputs(tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB reference checkout not available")

    values = np.linspace(-1.0, 1.0, 101)
    output = tmp_path / "rsfit_outputs.mat"
    engine = matlab_engine.start_matlab()
    try:
        engine.addpath(str(eeglab_root / "functions" / "timefreqfunc"), nargout=0)
        engine.eval(
            f"""
            x = [{_matlab_vector(values)}];
            [pval,c,l] = rsfit(x, 0, 0);
            solvres = rspdfsolv([1 1], 0, 1.8);
            [a1,a2,a3,a4] = rsadjust(1, 1, 0, 1/12, 0);
            getp = rsget([0 2 1 1], 0.25);
            funcres = rspfunc(0.75, [0 2 1 1], 0.25);
            save('{_matlab_string(output)}', 'pval', 'c', 'l', 'solvres', 'a1', 'a2', 'a3', 'a4', 'getp', 'funcres');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    matlab = scipy.io.loadmat(output, squeeze_me=True)
    py_pvalue, py_cumulants, py_lambdas, _chi2 = rsfit(values, 0.0, return_details=True)

    np.testing.assert_allclose(py_pvalue, matlab["pval"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(py_cumulants, matlab["c"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(py_lambdas, matlab["l"], rtol=1e-5, atol=1e-5)
    assert rspdfsolv([1.0, 1.0], 0.0, 1.8) == pytest.approx(float(matlab["solvres"]), abs=1e-12)
    np.testing.assert_allclose(
        rsadjust(1.0, 1.0, 0.0, 1.0 / 12.0, 0.0), [matlab["a1"], matlab["a2"], matlab["a3"], matlab["a4"]]
    )
    assert rsget([0.0, 2.0, 1.0, 1.0], 0.25) == pytest.approx(float(matlab["getp"]), abs=1e-8)
    assert rspfunc(0.75, [0.0, 2.0, 1.0, 1.0], 0.25) == pytest.approx(float(matlab["funcres"]), abs=1e-12)


def _assert_python_command(command: str) -> None:
    ast.parse(command, mode="eval")


def _matlab_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.17g}" for value in np.asarray(values, dtype=float).ravel())


def _matlab_string(path: Path) -> str:
    return str(path).replace("'", "''")


def _eeglab_reference_root() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    if os.environ.get("EEGPREP_EEGLAB_ROOT"):
        candidates.append(Path(os.environ["EEGPREP_EEGLAB_ROOT"]))
    candidates.extend(
        [
            repo_root / "src" / "eegprep" / "eeglab",
            repo_root.parent / "eeglab",
            Path("/tmp/eeglab-timefreq-ref"),
        ]
    )
    for candidate in candidates:
        if (candidate / "functions" / "sigprocfunc" / "signalstat.m").exists():
            return candidate
    return None
