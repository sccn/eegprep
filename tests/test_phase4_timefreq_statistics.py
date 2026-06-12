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

import eegprep
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher, action_kind
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_eventstat import event_values, pop_eventstat, pop_eventstat_dialog_spec
from eegprep.functions.popfunc.pop_crossf import pop_crossf
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_newcrossf import pop_newcrossf, pop_newcrossf_dialog_spec
from eegprep.functions.popfunc.pop_newtimef import _run_gui as run_newtimef_gui
from eegprep.functions.popfunc.pop_newtimef import pop_newtimef, pop_newtimef_dialog_spec
from eegprep.functions.popfunc.pop_timef import pop_timef
from eegprep.functions.popfunc.pop_signalstat import pop_signalstat, pop_signalstat_dialog_spec
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
from eegprep.functions.timefreqfunc.correct_mc import correct_mc
from eegprep.functions.timefreqfunc.correctfit import correctfit
from eegprep.functions.timefreqfunc.dftfilt import dftfilt
from eegprep.functions.timefreqfunc.dftfilt2 import dftfilt2
from eegprep.functions.timefreqfunc.dftfilt3 import dftfilt3
from eegprep.functions.statistics.fdr import fdr
from eegprep.functions.statistics.stat_surrogate_pvals import stat_surrogate_pvals
from eegprep.functions.timefreqfunc.newcrossf import _is_on as newcrossf_is_on
from eegprep.functions.timefreqfunc.newcrossf import _threshold_vector as newcrossf_threshold_vector
from eegprep.functions.timefreqfunc.newcrossf import _upper_thresholds_by_frequency
from eegprep.functions.timefreqfunc.newcrossf import newcrossf
from eegprep.functions.timefreqfunc._pac_support import _empirical_pvalue as pac_empirical_pvalue
from eegprep.functions.timefreqfunc.newtimef import _is_on as newtimef_is_on
from eegprep.functions.timefreqfunc.newtimef import _significance_mask, _thresholds_by_frequency
from eegprep.functions.timefreqfunc.newtimef import _threshold_vector as newtimef_threshold_vector
from eegprep.functions.timefreqfunc.newtimef import compute_time_frequency, newtimef
from eegprep.functions.timefreqfunc.newtimefbaseln import newtimefbaseln
from eegprep.functions.timefreqfunc.newtimefpowerunit import newtimefpowerunit
from eegprep.functions.timefreqfunc.rsadjust import rsadjust
from eegprep.functions.timefreqfunc.rsfit import rsfit
from eegprep.functions.timefreqfunc.rsget import rsget
from eegprep.functions.timefreqfunc.rspdfsolv import rspdfsolv
from eegprep.functions.timefreqfunc.rspfunc import rspfunc
from eegprep.functions.timefreqfunc.tf_cycle_calc import tf_cycle_calc
from eegprep.functions.timefreqfunc.timefreq import timefreq
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


def test_newtimef_synthetic_returns_deterministic_shapes():
    srate = 128
    times = np.arange(0, 1, 1 / srate)
    trials = np.stack([np.sin(2 * np.pi * 10 * times), np.sin(2 * np.pi * 10 * times + 0.2)], axis=1)

    result = newtimef(trials, trials.shape[0], [0, 1000], srate, 0, freqs=[5, 20], timesout=12, plot="off")

    assert result.ersp.shape == result.itc.shape == (result.freqs.size, result.times.size)
    assert result.tfdata.shape[2] == trials.shape[1]
    assert result.times.size <= 12
    assert np.isfinite(result.ersp).all()
    assert np.all(np.abs(result.itc) <= 1 + 1e-12)


def test_newtimef_rejects_unknown_options():
    signal = np.sin(2 * np.pi * 10 * np.arange(128) / 128)

    with pytest.raises(TypeError, match="unexpected keyword"):
        newtimef(signal, 128, [0, 1000], 128, 0, unsupported_option=1)


def test_newtimef_is_on_uses_whitelist_semantics():
    assert newtimef_is_on("on") is True
    assert newtimef_is_on("yes") is True
    assert newtimef_is_on(1) is True
    # Unrecognized values are treated as OFF, matching the canonical is_on.
    assert newtimef_is_on("yes-please") is False
    assert newtimef_is_on("display") is False
    assert newtimef_is_on("off") is False
    assert newtimef_is_on([0, 1]) is False
    assert newtimef_is_on(np.array([0, 1])) is False


def test_newtimef_fails_loudly_on_unimplemented_overlap_and_plotphase():
    signal = np.sin(2 * np.pi * 10 * np.arange(128) / 128)

    with pytest.raises(NotImplementedError, match="overlap"):
        newtimef(signal, 128, [0, 1000], 128, 0, plot="off", overlap=2)
    with pytest.raises(NotImplementedError, match="plotphase"):
        newtimef(signal, 128, [0, 1000], 128, 0, plot="off", plotphase="on")
    with pytest.raises(NotImplementedError, match="overlap"):
        compute_time_frequency(signal, 128, [0, 1000], 128, 0, overlap=2)

    # Default values still compute without raising.
    result = newtimef(signal, 128, [0, 1000], 128, 0, plot="off", overlap=None, plotphase="off")
    assert result.ersp.shape == result.itc.shape


def test_newtimef_nonzero_cycles_use_wavelet_time_grid(sample_epoch):
    result = pop_newtimef(sample_epoch, 1, 1, [-100, 200], [3, 0.8], plot="off")

    assert result.times.size > 1
    assert result.freqs.size > 0
    assert result.tfdata.shape == (result.freqs.size, result.times.size, sample_epoch["trials"])


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


def test_timewarp_rejects_unsorted_markers():
    with pytest.raises(ValueError, match="ascending order"):
        timewarp([1, 5, 3], [1, 2, 5])


def test_angtimewarp_interpolates_and_wraps_like_eeglab():
    angles = np.asarray([0, np.pi / 2, np.pi, -np.pi / 2, 0], dtype=float)

    warped = angtimewarp([1, 3, 5], [1, 2, 5], angles)

    np.testing.assert_allclose(warped, [0, np.pi, 0, -np.pi / 3, 0], rtol=1e-12, atol=1e-12)


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


def test_newcrossf_rejects_unknown_options():
    signal = np.sin(2 * np.pi * 10 * np.arange(128) / 128)

    with pytest.raises(TypeError, match="Unsupported newcrossf option"):
        newcrossf(signal, signal, 128, [0, 1000], 128, 0, unsupported_option=1, plot="off")


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


def test_pop_eventstat_epoched_latrange_uses_epoch_relative_latency(sample_epoch):
    all_values = event_values(sample_epoch, "latency", type=["square"])
    ranged_values = event_values(sample_epoch, "latency", type=["square"], latrange=[-100, 200])

    assert ranged_values.size == all_values.size == sample_epoch["trials"]


def test_pop_newtimef_gui_defaults_are_replayable_and_honest(sample_eeg):
    result = run_newtimef_gui(sample_eeg, typeproc=1, renderer=_DefaultDialogRenderer())

    assert result["options"]["timesout"] == 200
    assert "plotphase" not in result["options"]
    assert "plottype" not in result["options"]


def test_pop_newtimef_baseline_modes_bootstrap_and_curve_plot(sample_epoch):
    result, command = pop_newtimef(
        sample_epoch,
        1,
        1,
        [-100, 200],
        [3, 0.8],
        basenorm="on",
        trialbase="full",
        alpha=0.1,
        naccu=12,
        rng=0,
        plottype="curve",
        plot="off",
        return_com=True,
    )

    assert result.ersp.shape == result.itc.shape
    assert result.erspboot.shape == (result.freqs.size, 2)
    assert result.itcboot.shape == (result.freqs.size,)
    assert result.ersp_significant.shape == result.ersp.shape
    assert "basenorm='on'" in command
    assert "trialbase='full'" in command
    _assert_python_command(command)


def test_pop_newtimef_accepts_supplied_bootstrap_limits(sample_epoch):
    result = pop_newtimef(
        sample_epoch,
        1,
        1,
        [-100, 200],
        [0],
        alpha=0.05,
        pboot=np.asarray([[-1, 1], [-1, 1]], dtype=float),
        rboot=np.asarray([0.5, 0.5], dtype=float),
        freqs=[8, 12],
        nfreqs=2,
        plot="off",
    )

    np.testing.assert_array_equal(result.erspboot, np.asarray([[-1, 1], [-1, 1]], dtype=float))
    np.testing.assert_array_equal(result.itcboot, np.asarray([0.5, 0.5], dtype=float))


def test_timefreq_statistics_dialog_specs_match_eeglab_control_inventory(sample_eeg):
    newtimef = pop_newtimef_dialog_spec(sample_eeg, typeproc=1)
    newcrossf = pop_newcrossf_dialog_spec(sample_eeg, typeproc=1)
    signal_spec = pop_signalstat_dialog_spec(sample_eeg, typeproc=1)
    event_spec = pop_eventstat_dialog_spec(sample_eeg)

    assert newtimef.title == "Plot channel time frequency -- pop_newtimef()"
    assert newtimef.size == (1059, 511)
    assert newtimef.row_spacing == 4
    assert newcrossf.size == (908, 476)
    assert newcrossf.row_spacing == 4
    assert controls_by_tag(newtimef)["num_button"].callback.name == "select_channels"
    assert controls_by_tag(newtimef)["baseline"].value == "0"
    assert controls_by_tag(newtimef)["calcpush"].enabled is True
    assert controls_by_tag(newtimef)["calcpush"].callback.name == "tf_cycle_calc"
    assert controls_by_tag(newtimef)["plotcurve"].enabled is True
    assert controls_by_tag(newtimef)["alpha"].enabled is True
    assert controls_by_tag(newcrossf)["coher"].value is False
    assert signal_spec.title == "Plot signal statistics -- pop_signalstat()"
    assert controls_by_tag(signal_spec)["percent"].value == "5"
    assert event_spec.title == "Plot event statistics -- pop_eventstat()"
    assert controls_by_tag(event_spec)["eventfield"].value == "latency"


def test_timefreq_statistics_menu_actions_are_implemented():
    assert action_kind("pop_newtimef:channels") == "implemented"
    assert action_kind("pop_newcrossf:components") == "implemented"
    assert action_kind("pop_signalstat:channels") == "implemented"
    assert action_kind("pop_eventstat") == "implemented"


def test_phase4_top_level_exports_resolve_existing_modules():
    expected = {
        "bootstat",
        "correct_mc",
        "crossf",
        "dftfilt",
        "dftfilt2",
        "dftfilt3",
        "angtimewarp",
        "newtimefbaseln",
        "newtimefitc",
        "newtimefpowerunit",
        "newtimeftrialbaseln",
        "pop_crossf",
        "pop_timef",
        "tf_cycle_calc",
        "timef",
        "timefreq",
        "timewarp",
    }
    assert expected <= set(eegprep.__all__)
    for name in expected:
        assert getattr(eegprep, name) is not None


def test_legacy_timef_and_crossf_wrappers_return_replayable_history(sample_epoch):
    timef_result, timef_command = pop_timef(sample_epoch, 1, 1, [-100, 200], [0], plot="off", return_com=True)
    crossf_result, crossf_command = pop_crossf(sample_epoch, 1, 1, 2, [-100, 200], [0], plot="off", return_com=True)

    assert timef_result.ersp.ndim == 2
    assert crossf_result.coherence.ndim == 2
    assert timef_command.startswith("pop_timef(EEG, 1, 1")
    assert crossf_command.startswith("pop_crossf(EEG, 1, 1, 2")
    _assert_python_command(timef_command)
    _assert_python_command(crossf_command)


def test_newcrossf_supported_phase4_options_are_headless():
    srate = 128
    times = np.arange(0, 1, 1 / srate)
    trials = np.stack(
        [
            np.sin(2 * np.pi * 12 * times),
            np.sin(2 * np.pi * 12 * times + 0.3),
            np.sin(2 * np.pi * 12 * times + 0.7),
        ],
        axis=1,
    )

    result = newcrossf(
        trials,
        trials,
        trials.shape[0],
        [0, 1000],
        srate,
        0,
        freqs=[8, 16],
        type="phasecoher2",
        subitc="on",
        alpha=0.1,
        naccu=12,
        rng=0,
        plot="off",
    )
    amp = newcrossf(trials, trials, trials.shape[0], [0, 1000], srate, 0, freqs=[8, 16], type="amp", plot="off")

    assert result.rboot.shape == (result.freqs.size,)
    assert result.significant.shape == result.coherence.shape
    assert np.nanmax(np.abs(result.coherence)) <= 1 + 1e-12
    assert amp.coherence.shape == amp.phase.shape


def test_newtimefpowerunit_matches_eeglab_labels():
    assert newtimefpowerunit({"scale": "log", "baseline": 0, "basenorm": "off"}) == "dB"
    assert newtimefpowerunit({"scale": "abs", "baseline": [float("nan")], "basenorm": "off"}) == "uV^{2}/Hz"
    assert newtimefpowerunit({"scale": "abs", "baseline": 0, "basenorm": "on"}) == "std."


def test_bootstat_threshold_sides_and_rand_phase_preserve_magnitude():
    first = np.arange(24, dtype=float).reshape(2, 3, 4)
    second = first[:, ::-1, :] / 2.0

    def statistic(left, right):
        return np.mean(left - right, axis=1)

    upper = bootstat([first, second], statistic=statistic, alpha=0.1, naccu=20, bootside="upper", rng=0)
    both = bootstat([first, second], statistic=statistic, alpha=0.1, naccu=20, bootside="both", rng=0)

    assert upper.surrogates.shape == (20, 2, 4)
    assert upper.thresholds.shape == (2, 4)
    assert both.thresholds.shape == (2, 4, 2)
    np.testing.assert_allclose(upper.thresholds, bootstrap_threshold(upper.surrogates, alpha=0.1, bootside="upper"))

    complex_values = np.exp(1j * np.arange(6, dtype=float)).reshape(2, 3)
    randomized = bootstat(complex_values, statistic=np.abs, naccu=5, boottype="rand", rng=0)
    np.testing.assert_allclose(randomized.surrogates, np.broadcast_to(np.abs(complex_values), (5, 2, 3)))


def test_timefreq_threshold_helpers_pool_through_canonical_bootstrap_threshold():
    # newtimef/newcrossf no longer re-sort surrogates; they pool (naccu x baseline)
    # per frequency and delegate the percentile/tail math to bootstrap_threshold.
    rng = np.random.default_rng(7)
    surrogates = rng.normal(size=(24, 3, 5))

    pooled = surrogates.transpose(0, 2, 1).reshape(-1, surrogates.shape[1])
    expected_both = bootstrap_threshold(pooled, alpha=0.1, bootside="both")
    expected_upper = bootstrap_threshold(pooled, alpha=0.1, bootside="upper")

    np.testing.assert_allclose(_thresholds_by_frequency(surrogates, alpha=0.1, both=True), expected_both)
    np.testing.assert_allclose(_thresholds_by_frequency(surrogates, alpha=0.1, both=False), expected_upper)
    np.testing.assert_allclose(_upper_thresholds_by_frequency(surrogates, alpha=0.1), expected_upper)

    # Single-frequency case keeps the original (nfreq,) / (nfreq, 2) shapes.
    single = rng.normal(size=(24, 1, 5))
    assert _thresholds_by_frequency(single, alpha=0.1, both=True).shape == (1, 2)
    assert _thresholds_by_frequency(single, alpha=0.1, both=False).shape == (1,)
    assert _upper_thresholds_by_frequency(single, alpha=0.1).shape == (1,)


def test_timefreq_shared_bootstrap_helpers_cover_newtimef_and_newcrossf_paths():
    times = np.asarray([-100.0, 0.0, 100.0, 200.0])
    baseln = np.asarray([0, 1], dtype=int)

    np.testing.assert_array_equal(bootstrap_indices(times, baseline=0, baseboot=[], baseln=baseln), baseln)
    np.testing.assert_array_equal(bootstrap_indices(times, baseline=np.nan, baseboot=1, baseln=baseln), [0, 1])
    np.testing.assert_array_equal(bootstrap_indices(times, baseline=np.nan, baseboot=0, baseln=baseln), [])
    np.testing.assert_array_equal(bootstrap_indices(times, baseline=np.nan, baseboot=[50, 200], baseln=baseln), [2, 3])
    np.testing.assert_array_equal(bootstrap_indices(times, baseboot=1, baseln=None, limit_to_baseboot=True), [0, 1])

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


def test_newtimef_fdr_branch_matches_canonical_fdr_threshold():
    rng = np.random.default_rng(11)
    pvalues = rng.random(size=(4, 6))

    threshold = float(fdr(pvalues, 0.1).threshold)
    expected = np.zeros_like(pvalues, dtype=bool) if threshold == 0 else pvalues <= threshold

    np.testing.assert_array_equal(_significance_mask(pvalues, 0.1, "fdr"), expected)
    np.testing.assert_array_equal(_significance_mask(pvalues, 0.1, "none"), pvalues <= 0.1)


def test_timefreq_is_on_and_threshold_vector_are_the_canonical_shared_helpers():
    # newcrossf reuses newtimef's threshold helper and the canonical is_on whitelist.
    assert newcrossf_threshold_vector is newtimef_threshold_vector
    assert newtimef_threshold_vector is threshold_vector
    assert newcrossf_is_on is newtimef_is_on
    assert newcrossf_is_on("on") is True
    assert newcrossf_is_on("display") is False


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


def test_correct_mc_returns_phase4_standalone_shapes():
    rng = np.random.default_rng(0)
    eeg = {
        "data": rng.normal(size=(2, 64, 4)),
        "pnts": 64,
        "trials": 4,
        "srate": 128,
        "xmin": 0,
        "xmax": 63 / 128,
    }

    ncorrect, pvalues = correct_mc(eeg, cycles=0, freqrange=(4, 16), timesout=(4, 6))

    assert isinstance(ncorrect, int)
    assert pvalues.shape == (int(np.ceil(np.log2(16))), 2)


def test_correct_mc_uses_rsfit_for_neighbor_correlations():
    eeg = {
        "data": np.arange(64, dtype=float).reshape(2, 32),
        "pnts": 32,
        "srate": 128,
        "xmin": 0,
        "xmax": 31 / 128,
    }

    class Result:
        def __init__(self, offset):
            base = np.asarray(
                [
                    [0.0, 1.0, 2.0, 4.0],
                    [0.0, 1.0, 3.0, 6.0],
                    [0.0, 2.0, 5.0, 9.0],
                ],
                dtype=float,
            )
            self.ersp = base + offset

    def fake_newtimef(data, *_args, **_kwargs):
        return Result(float(data[0]))

    correct_mc_module = importlib.import_module("eegprep.functions.timefreqfunc.correct_mc")
    with (
        mock.patch.object(correct_mc_module, "newtimef", side_effect=fake_newtimef),
        mock.patch.object(correct_mc_module, "rsfit", return_value=0.001) as fitted,
    ):
        ncorrect, pvalues = correct_mc(eeg, cycles=0, freqrange=(4, 8), timesout=(4,))

    assert ncorrect == 12
    assert fitted.call_count == 3
    assert pvalues.shape == (3, 1)
    for call in fitted.call_args_list:
        correlations, value = call.args
        assert value == 0.0
        assert len(correlations) == 2


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


def test_legacy_dft_helpers_cover_public_surface():
    filters = dftfilt(16, 2, 4, 2, 0.5)
    empty = dftfilt(8, 0.1, 10, 2, 0.5)
    wavelets = dftfilt2([8, 16], [3, 5], 128)
    sinus = dftfilt2([8], 3, 128, kind="sinus")

    assert filters.shape == (16, 14)
    assert np.iscomplexobj(filters)
    assert empty.shape == (8, 0)
    assert [wavelet.size for wavelet in wavelets] == [49, 41]
    assert sinus[0].shape == (49,)
    assert np.iscomplexobj(sinus[0])


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
        (
            "pop_signalstat:channels",
            "eegprep.functions.popfunc.pop_signalstat.pop_signalstat",
            {"typeproc": 1, "return_com": True},
            "pop_signalstat(EEG, 1, 1, 5)",
        ),
        (
            "pop_eventstat",
            "eegprep.functions.popfunc.pop_eventstat.pop_eventstat",
            {"return_com": True},
            "pop_eventstat(EEG, 'latency', [], [], 5)",
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


def test_signalstat_plots_topographic_context_when_map_is_available(ica_epoch):
    result = pop_signalstat(ica_epoch, 0, 1, 5)

    titles = [axis.get_title() for axis in result.figure.axes]
    assert "Topographic map" in titles
    plt.close(result.figure)


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


class _DefaultDialogRenderer:
    def run(self, spec, initial_values=None):
        _ = initial_values
        return {control.tag: control.value for control in spec.controls if control.tag}
