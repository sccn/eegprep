"""Ports of the maintained grouped STUDY measure-plot wrapper methods.

Upstream suite: sccn/eeglab_tests@ff605546f3f70868916fb8d49c007472b3257b50
EEGLAB tree: sccn/eeglab@8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pytest
from scipy.fft import fft
from scipy.signal import detrend
from scipy.stats import ttest_rel

from eegprep import (
    pop_statparams,
    pop_study,
    std_erpplot,
    std_erspplot,
    std_itcplot,
    std_precomp,
    std_specplot,
    std_stat,
    std_topoplot,
)
from tests.eeglab_tests import eeglab_test, load_matlab_test_fixture


STUDYFUNC_ROOT = "unittesting_studyfunc"


def _reference(wrapper: str, test: str):
    source = f"{STUDYFUNC_ROOT}/{wrapper}/studyfunc_{wrapper}_wrapperTest.m"
    return eeglab_test(source, test)


def _records(value):
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return [{field: record[field] for field in value.dtype.names} for record in value.ravel(order="F")]


def _cell_row(*values):
    cells = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        cells[0, index] = value
    return cells


def _read_measure_study(backend, directory, filename="n400clustedit.study"):
    study, alleeg = backend("pop_loadstudy", filename=filename, filepath=str(directory), nargout=2)
    study = backend("std_checkset", study, alleeg)
    assert Path(study["filepath"]).is_relative_to(directory)
    assert all(Path(eeg["filepath"]).is_relative_to(directory) for eeg in _records(alleeg))
    return study, alleeg


def _close_measure_plot(backend, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend("close", nargout=0)
    else:
        plt.close()


def _assert_measure(actual, expected, tolerance=1e-4, sign_tolerance=None):
    # These source tests define their own strict absolute assertions, not near.m.
    difference = np.all(np.abs(np.asarray(actual) - expected) < tolerance)
    if sign_tolerance is not None:
        difference = difference or np.all(np.abs(np.asarray(actual) + expected) < sign_tolerance)
    assert difference, (
        f"Maximum absolute difference {np.max(np.abs(np.asarray(actual) - expected))}; "
        f"tolerance {tolerance}; shapes {np.shape(actual)}, {np.shape(expected)}"
    )


def _text(value):
    return str(np.asarray(value).item())


def _strmatch(value, cells):
    return np.array(
        [index for index, cell in enumerate(np.asarray(cells, dtype=object).ravel(order="F")) if _text(cell) == value],
        dtype=int,
    )


def _measure_channels(backend, alleeg):
    locations = backend("eeg_mergelocs", *(eeg["chanlocs"] for eeg in _records(alleeg)))
    channels = _records(locations)
    selected = np.random.default_rng(1).integers(len(channels))
    return locations, channels[selected]["labels"]


def _channel_oracle_eeg(backend, alleeg, locations):
    # The pinned reference selects EEGLAB's modern (>14) source branch:
    # datInds is 1, with all trials of ALLEEG(1), not a random legacy cell.
    eeg = _records(alleeg)[0].copy()
    trials = np.arange(1.0, np.asarray(eeg["trials"]).item() + 1)[None, :]
    eeg["data"] = backend("eeg_getdatact", eeg, "trialindices", trials)
    eeg["trials"] = float(eeg["data"].shape[2] if eeg["data"].ndim > 2 else 1)
    eeg["epoch"] = np.empty((0, 0))
    if np.asarray(eeg["nbchan"]).item() < len(_records(locations)):
        eeg = backend("eeg_interp", eeg, locations)
    return eeg


def _channel_case(study, design_index):
    first = _records(study["datasetinfo"])[0]
    cases = _records(_records(study["design"])[design_index]["cases"])[0]["value"]
    indices = _strmatch(first["subject"], cases)
    return indices, Path(first["filepath"]) / first["subject"]


def _component_case(study, alleeg, design_index, component_index, condition_index):
    cluster = _records(study["cluster"])[2]
    dataset_index = int(cluster["sets"][condition_index, component_index]) - 1
    component = np.asarray(cluster["comps"]).ravel(order="F")[component_index]
    info = _records(study["datasetinfo"])[dataset_index]
    variable = _records(_records(study["design"])[design_index]["variable"])[0]
    condition_positions = _strmatch(info["condition"], variable["value"])
    eeg = _records(alleeg)[dataset_index].copy()
    return eeg, component, info["condition"], int(condition_positions.item()), Path(info["filepath"]) / info["subject"]


def _cached_trials(cache, condition):
    return np.array(
        [index for index, trial in enumerate(_records(cache["trialinfo"])) if _text(trial["condition"]) == condition],
        dtype=int,
    )


def _spectral_oracle(backend, request, data, srate):
    # test_stdspecplot3/4: per-trial detrend, symmetric hamming2, FFT,
    # remove DC, then mean the single-trial log powers (not log mean power).
    # MATLAB's single-precision FFT and SciPy's differ beyond the source's
    # strict 1e-4 assertion. Validate the MATLAB lane with its own primitives.
    native = request.config.getoption("--eeglab-backend") == "matlab"
    detrended = backend("eegprep_test_detrend_trials", data) if native else detrend(data, axis=1)
    points = data.shape[1]
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(points) / (points - 1))
    # MATLAB rounds the mixed single/double product, not the double window.
    windowed = (detrended * window.reshape((1, points) + (1,) * (data.ndim - 2))).astype(data.dtype)
    transformed = backend("fft", windowed, np.empty((0, 0)), 2.0) if native else fft(windowed, axis=1)
    frequencies = np.linspace(0, srate / 2, points // 2)[None, 1:]
    power = 10 * np.log10(np.abs(transformed[:, 1 : points // 2]) ** 2)
    return power.mean(axis=2) if power.ndim > 2 else power, frequencies


def _timef_cache_options(cache):
    entries = cache["parameters"].ravel(order="F")
    parameters = {_text(entries[index]): entries[index + 1] for index in range(0, len(entries), 2)}
    cycles = parameters.pop("cycles")
    return cycles, tuple(value for pair in parameters.items() for value in pair)


def _time_limits(eeg):
    return np.array([[np.asarray(eeg["xmin"]).item(), np.asarray(eeg["xmax"]).item()]]) * 1000


@pytest.mark.slow
@_reference("std_erspplot", "test_test_std_erspplot2_2")
def test_reference_std_erspplot_channel_cache(eeglab_backend, eeglab_writable_study, request):
    study, alleeg = _read_measure_study(eeglab_backend, eeglab_writable_study)
    locations, channel = _measure_channels(eeglab_backend, alleeg)
    cycles = np.array([[3.0, 0.8]])
    for design_index in range(len(_records(study["design"]))):
        cases, filebase = _channel_case(study, design_index)
        trials = int(np.asarray(_records(alleeg)[0]["trials"]).item())
        study = eeglab_backend("std_selectdesign", study, alleeg, float(design_index + 1))
        parameters = _cell_row(
            "cycles", cycles, "nfreqs", 10.0, "ntimesout", 10.0, "baseline", np.nan, "verbose", "off"
        )
        study, alleeg = eeglab_backend(
            "std_precomp",
            study,
            alleeg,
            "channels",
            "savetrials",
            "on",
            "recompute",
            "on",
            "interp",
            "on",
            "ersp",
            "on",
            "erspparams",
            parameters,
            nargout=2,
        )
        study, cells, times, frequencies = eeglab_backend(
            "std_erspplot", study, alleeg, "channels", _cell_row(channel), nargout=4
        )
        _close_measure_plot(eeglab_backend, request)
        values = cells[1, 0]
        if values.size and cases.size and np.all(cases < values.shape[1]):
            plotted = values[:, :, 0, int(cases.item())]
            cache = load_matlab_test_fixture(str(filebase) + ".dattimef")
            channel_index = int(_strmatch(channel, cache["labels"]).item())
            cached_cycles, cached_options = _timef_cache_options(cache)
            precomputed = {
                "tfdata": cache[f"chan{channel_index + 1}"][:, :, :trials],
                "times": times,
                "freqs": frequencies,
                "recompute": "ersp",
            }
            first = _records(alleeg)[0]
            cached = eeglab_backend(
                "newtimef",
                np.empty((0, 0)),
                first["pnts"],
                _time_limits(first),
                1.0,
                cached_cycles,
                "precomputed",
                precomputed,
                "verbose",
                "off",
                *cached_options,
            )
            eeg = _channel_oracle_eeg(eeglab_backend, alleeg, locations)
            labels = _cell_row(*(location["labels"] for location in _records(eeg["chanlocs"])))
            channel_index = int(_strmatch(channel, labels).item())
            frequencies_arg = np.array([[3.0, np.asarray(eeg["srate"]).item() / 2]])
            options = (
                "freqscale",
                "log",
                "nfreqs",
                10.0,
                "ntimesout",
                10.0,
                "baseline",
                np.nan,
                "freqs",
                frequencies_arg,
            )
            recomputed, _itc, _base, oracle_times, oracle_frequencies, _eboot, _iboot, coefficients = eeglab_backend(
                "newtimef",
                eeg["data"][channel_index : channel_index + 1],
                eeg["pnts"],
                _time_limits(eeg),
                eeg["srate"],
                cycles,
                *options,
                nargout=8,
            )
            _close_measure_plot(eeglab_backend, request)
            precomputed = {
                "tfdata": coefficients,
                "times": oracle_times,
                "freqs": oracle_frequencies,
                "recompute": "ersp",
            }
            cached_recomputed, _itc, _base, oracle_times, oracle_frequencies, _eboot, _iboot, _coefficients = (
                eeglab_backend(
                    "newtimef",
                    np.empty((0, 0)),
                    eeg["pnts"],
                    _time_limits(eeg),
                    eeg["srate"],
                    cycles,
                    "precomputed",
                    precomputed,
                    "plotersp",
                    "off",
                    "plotitc",
                    "off",
                    *options,
                    nargout=8,
                )
            )
            _close_measure_plot(eeglab_backend, request)
            _assert_measure(times, oracle_times, tolerance=1e-3)
            _assert_measure(frequencies, oracle_frequencies, tolerance=1e-3)
            _assert_measure(cached, plotted, tolerance=1e-3)
            _assert_measure(cached, recomputed, tolerance=1e-3)
            _assert_measure(cached, cached_recomputed, tolerance=1e-3)


@pytest.mark.slow
@_reference("std_erspplot", "test_test_std_erspplot3_2")
def test_reference_std_erspplot_component_cache(eeglab_backend, eeglab_writable_study, request):
    study, alleeg = _read_measure_study(eeglab_backend, eeglab_writable_study)
    _measure_channels(eeglab_backend, alleeg)
    cycles = np.array([[3.0, 0.8]])
    for design_index in range(len(_records(study["design"]))):
        study = eeglab_backend("std_selectdesign", study, alleeg, float(design_index + 1))
        parameters = _cell_row(
            "cycles", cycles, "nfreqs", 10.0, "ntimesout", 10.0, "baseline", np.nan, "verbose", "off"
        )
        study, alleeg = eeglab_backend(
            "std_precomp",
            study,
            alleeg,
            "components",
            "savetrials",
            "on",
            "recompute",
            "on",
            "interp",
            "on",
            "ersp",
            "on",
            "erspparams",
            parameters,
            nargout=2,
        )
        study, cells, times, frequencies = eeglab_backend("std_erspplot", study, alleeg, "clusters", 3.0, nargout=4)
        _close_measure_plot(eeglab_backend, request)
        eeg, component, condition, condition_index, filebase = _component_case(study, alleeg, design_index, 1, 1)
        plotted = cells.ravel(order="F")[condition_index][:, :, 1]
        cache = load_matlab_test_fixture(str(filebase) + ".icatimef")
        cached_cycles, cached_options = _timef_cache_options(cache)
        precomputed = {
            "tfdata": cache[f"comp{int(component)}"][:, :, _cached_trials(cache, condition)],
            "times": times,
            "freqs": frequencies,
            "recompute": "ersp",
        }
        cached = eeglab_backend(
            "newtimef",
            np.empty((0, 0)),
            eeg["pnts"],
            _time_limits(eeg),
            1.0,
            cached_cycles,
            "precomputed",
            precomputed,
            "verbose",
            "off",
            *cached_options,
        )
        trials = np.arange(1.0, np.asarray(eeg["trials"]).item() + 1)[None, :]
        data = eeglab_backend("eeg_getdatact", eeg, "component", component, "trialindices", trials)
        recomputed, _itc, _base, oracle_times, _oracle_frequencies, _eboot, _iboot, _coefficients = eeglab_backend(
            "newtimef",
            data,
            eeg["pnts"],
            _time_limits(eeg),
            eeg["srate"],
            cycles,
            "freqscale",
            "log",
            "nfreqs",
            10.0,
            "ntimesout",
            10.0,
            "baseline",
            np.nan,
            "freqs",
            np.array([[3.0, np.asarray(eeg["srate"]).item() / 2]]),
            nargout=8,
        )
        _close_measure_plot(eeglab_backend, request)
        if np.sign(plotted.flat[0]) != np.sign(cached.flat[0]):
            plotted = -plotted
        _assert_measure(times, oracle_times, sign_tolerance=1e-4)
        _assert_measure(cached, plotted, sign_tolerance=1e-4)
        _assert_measure(cached, recomputed, sign_tolerance=1e-4)


@pytest.mark.slow
@_reference("std_erpplot", "test_test_stderpplot2")
def test_reference_std_erpplot_channel_cache(eeglab_backend, eeglab_writable_study, request):
    study, alleeg = _read_measure_study(eeglab_backend, eeglab_writable_study)
    locations, channel = _measure_channels(eeglab_backend, alleeg)
    for design_index in range(len(_records(study["design"]))):
        cases, filebase = _channel_case(study, design_index)
        trials = int(np.asarray(_records(alleeg)[0]["trials"]).item())
        study = eeglab_backend("std_selectdesign", study, alleeg, float(design_index + 1))
        study, alleeg = eeglab_backend(
            "std_precomp",
            study,
            alleeg,
            "channels",
            "savetrials",
            "on",
            "recompute",
            "on",
            "interp",
            "on",
            "erp",
            "on",
            "erpparams",
            _cell_row("rmbase", np.empty((0, 0))),
            nargout=2,
        )
        study, cells, times = eeglab_backend("std_erpplot", study, alleeg, "channels", _cell_row(channel), nargout=3)
        _close_measure_plot(eeglab_backend, request)
        values = cells[1, 0]
        if values.size and cases.size and np.all(cases < values.shape[1]):
            cache = load_matlab_test_fixture(str(filebase) + ".daterp")
            channel_index = int(_strmatch(channel, cache["labels"]).item())
            cached = cache[f"chan{channel_index + 1}"][:, :trials].mean(axis=1, keepdims=True).T
            plotted = values[:, cases].T
            eeg = _channel_oracle_eeg(eeglab_backend, alleeg, locations)
            labels = _cell_row(*(location["labels"] for location in _records(eeg["chanlocs"])))
            channel_index = int(_strmatch(channel, labels).item())
            recomputed = eeg["data"].mean(axis=2)[channel_index : channel_index + 1]
            _assert_measure(times, eeg["times"])
            _assert_measure(cached, plotted)
            _assert_measure(cached, recomputed)


@pytest.mark.slow
@_reference("std_erpplot", "test_test_stderpplot3")
def test_reference_std_erpplot_component_cache(eeglab_backend, eeglab_writable_study, request):
    study, alleeg = _read_measure_study(eeglab_backend, eeglab_writable_study)
    _measure_channels(eeglab_backend, alleeg)  # Source also merges/selects a channel here.
    for design_index in range(len(_records(study["design"]))):
        study = eeglab_backend("std_selectdesign", study, alleeg, float(design_index + 1))
        study, alleeg = eeglab_backend(
            "std_precomp",
            study,
            alleeg,
            "components",
            "savetrials",
            "on",
            "recompute",
            "on",
            "interp",
            "on",
            "scalp",
            "on",
            "erp",
            "on",
            "erpparams",
            _cell_row("rmbase", np.empty((0, 0))),
            nargout=2,
        )
        study, cells, times = eeglab_backend("std_erpplot", study, alleeg, "clusters", 3.0, nargout=3)
        _close_measure_plot(eeglab_backend, request)
        eeg, component, condition, condition_index, filebase = _component_case(study, alleeg, design_index, 1, 1)
        plotted = cells.ravel(order="F")[condition_index][:, 1:2].T
        cache = load_matlab_test_fixture(str(filebase) + ".icaerp")
        cached = cache[f"comp{int(component)}"][:, _cached_trials(cache, condition)].mean(axis=1, keepdims=True).T
        trials = np.arange(1.0, np.asarray(eeg["trials"]).item() + 1)[None, :]
        data = eeglab_backend("eeg_getdatact", eeg, "component", component, "trialindices", trials)
        recomputed = data.mean(axis=2)
        if np.sign(plotted.flat[0]) != np.sign(cached.flat[0]):
            plotted = -plotted
        _assert_measure(times, eeg["times"], sign_tolerance=1e-4)
        _assert_measure(cached, plotted, sign_tolerance=1e-4)
        _assert_measure(cached, recomputed, sign_tolerance=1e-4)


@pytest.mark.slow
@_reference("std_specplot", "test_test_stdspecplot3")
def test_reference_std_specplot_channel_cache(eeglab_backend, eeglab_writable_study, request):
    study, alleeg = _read_measure_study(eeglab_backend, eeglab_writable_study)
    locations, channel = _measure_channels(eeglab_backend, alleeg)
    for design_index in range(len(_records(study["design"]))):
        cases, filebase = _channel_case(study, design_index)
        trials = int(np.asarray(_records(alleeg)[0]["trials"]).item())
        study = eeglab_backend("std_selectdesign", study, alleeg, float(design_index + 1))
        study, alleeg = eeglab_backend(
            "std_precomp",
            study,
            alleeg,
            "channels",
            "savetrials",
            "on",
            "recompute",
            "on",
            "interp",
            "on",
            "spec",
            "on",
            "specparams",
            _cell_row("specmode", "fft"),
            nargout=2,
        )
        study, cells, frequencies = eeglab_backend(
            "std_specplot", study, alleeg, "channels", _cell_row(channel), nargout=3
        )
        _close_measure_plot(eeglab_backend, request)
        values = cells[1, 0]
        if values.size and cases.size and np.all(cases < values.shape[1]):
            cache = load_matlab_test_fixture(str(filebase) + ".datspec")
            channel_index = int(_strmatch(channel, cache["labels"]).item())
            cached = cache[f"chan{channel_index + 1}"][:, :trials].mean(axis=1, keepdims=True).T
            plotted = values[:, cases].T
            eeg = _channel_oracle_eeg(eeglab_backend, alleeg, locations)
            recomputed, oracle_frequencies = _spectral_oracle(
                eeglab_backend, request, eeg["data"], np.asarray(eeg["srate"]).item()
            )
            labels = _cell_row(*(location["labels"] for location in _records(eeg["chanlocs"])))
            channel_index = int(_strmatch(channel, labels).item())
            _assert_measure(frequencies, oracle_frequencies)
            _assert_measure(cached, plotted)
            _assert_measure(cached, recomputed[channel_index : channel_index + 1])


@pytest.mark.slow
@_reference("std_specplot", "test_test_stdspecplot4")
def test_reference_std_specplot_component_cache(eeglab_backend, eeglab_writable_study, request):
    study, alleeg = _read_measure_study(eeglab_backend, eeglab_writable_study)
    eeglab_backend("eeg_mergelocs", *(eeg["chanlocs"] for eeg in _records(alleeg)))
    for design_index in range(len(_records(study["design"]))):
        study = eeglab_backend("std_selectdesign", study, alleeg, float(design_index + 1))
        study, alleeg = eeglab_backend(
            "std_precomp",
            study,
            alleeg,
            "components",
            "savetrials",
            "on",
            "recompute",
            "on",
            "spec",
            "on",
            "specparams",
            _cell_row("specmode", "fft"),
            nargout=2,
        )
        study, cells, frequencies = eeglab_backend("std_specplot", study, alleeg, "clusters", 3.0, nargout=3)
        _close_measure_plot(eeglab_backend, request)
        eeg, component, condition, condition_index, filebase = _component_case(study, alleeg, design_index, 4, 0)
        plotted = cells.ravel(order="F")[condition_index][:, 4:5].T
        cache = load_matlab_test_fixture(str(filebase) + ".icaspec")
        cached = cache[f"comp{int(component)}"][:, _cached_trials(cache, condition)].mean(axis=1, keepdims=True).T
        trials = np.arange(1.0, np.asarray(eeg["trials"]).item() + 1)[None, :]
        data = eeglab_backend("eeg_getdatact", eeg, "component", component, "trialindices", trials)
        recomputed, oracle_frequencies = _spectral_oracle(
            eeglab_backend, request, data, np.asarray(eeg["srate"]).item()
        )
        _assert_measure(frequencies, oracle_frequencies, tolerance=1e-3, sign_tolerance=1e-4)
        _assert_measure(cached, plotted, tolerance=1e-3, sign_tolerance=1e-4)
        _assert_measure(cached, recomputed, tolerance=1e-3, sign_tolerance=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("eeglab_writable_study", ["teststudy2"], indirect=True)
@_reference("std_erpplot", "test_test_std_erpplot")
def test_reference_std_erpplot_design_sweep(eeglab_backend, eeglab_writable_study, request):
    study, alleeg = _read_measure_study(eeglab_backend, eeglab_writable_study, "stern2s.study")
    locations = _records(_records(alleeg)[0]["chanlocs"])[2:9]
    channels = (_cell_row(*(location["labels"] for location in locations)), _cell_row("AF3"))
    together = ("apart", "together")
    stats = ("on", "off")
    effects = ("main", "marginal")
    for design_index in range(7, len(_records(study["design"]))):
        study = eeglab_backend("std_selectdesign", study, alleeg, float(design_index + 1))
        variables = _records(_records(study["design"])[design_index]["variable"])
        if len(variables) < 2:
            continue  # The original source explicitly omits single-variable designs.
        study, alleeg = eeglab_backend(
            "std_precomp",
            study,
            alleeg,
            _cell_row(),
            "interp",
            "on",
            "allcomps",
            "on",
            "recompute",
            "on",
            "erp",
            "on",
            nargout=2,
        )
        count = 2 if np.asarray(variables[1]["value"]).size > 1 else 1
        for channel in channels:
            for subjects in ("off", "on"):
                for group_layout in together[:count]:
                    for condition_layout in together[:count]:
                        for condition_stats in stats[:count]:
                            for _source_s2 in range(2):
                                # The source has two nested loops both named s2.
                                for group_index in range(count):
                                    options = (
                                        "channels",
                                        channel,
                                        "plotsubjects",
                                        subjects,
                                        "condstats",
                                        condition_stats,
                                        "groupstats",
                                        stats[group_index],
                                        "plotgroups",
                                        group_layout,
                                        "plotconditions",
                                        condition_layout,
                                    )
                                    study = eeglab_backend("std_erpplot", study, alleeg, *options, "threshold", np.nan)
                                    if condition_stats == "on" or stats[group_index] == "on":
                                        study = eeglab_backend(
                                            "std_erpplot",
                                            study,
                                            alleeg,
                                            "effect",
                                            effects[group_index],
                                            *options,
                                            "threshold",
                                            0.05,
                                        )
                                    _close_measure_plot(eeglab_backend, request)
                                    _close_measure_plot(eeglab_backend, request)


@pytest.mark.slow
@pytest.mark.parametrize("eeglab_writable_study", ["teststudy2"], indirect=True)
@_reference("std_specplot", "test_test_std_specplot2")
def test_reference_std_specplot_design_sweep(eeglab_backend, eeglab_writable_study, request):
    study, alleeg = _read_measure_study(eeglab_backend, eeglab_writable_study, "stern2s.study")
    locations = _records(_records(alleeg)[0]["chanlocs"])[2:9]
    channels = (_cell_row(*(location["labels"] for location in locations)), _cell_row("AF3"))
    together = ("apart", "together")
    stats = ("off", "on")
    for design_index in range(7, len(_records(study["design"]))):
        study = eeglab_backend("std_selectdesign", study, alleeg, float(design_index + 1))
        study, alleeg = eeglab_backend(
            "std_precomp",
            study,
            alleeg,
            _cell_row(),
            "interp",
            "on",
            "allcomps",
            "on",
            "recompute",
            "on",
            "spec",
            "on",
            "specparams",
            _cell_row("specmode", "fft"),
            nargout=2,
        )
        variables = _records(_records(study["design"])[design_index]["variable"])
        count = 2 if np.asarray(variables[1]["value"]).size > 1 else 1
        for channel in channels:
            for subjects in ("off", "on"):
                for group_layout in together[:count]:
                    for condition_layout in together[:count]:
                        for condition_stats in stats[:count]:
                            for group_stats in stats[:count]:
                                options = (
                                    "channels",
                                    channel,
                                    "plotsubjects",
                                    subjects,
                                    "condstats",
                                    condition_stats,
                                    "groupstats",
                                    group_stats,
                                    "plotgroups",
                                    group_layout,
                                    "plotconditions",
                                    condition_layout,
                                )
                                study = eeglab_backend("std_specplot", study, alleeg, *options, "threshold", np.nan)
                                if condition_stats == "on" or group_stats == "on":
                                    study = eeglab_backend("std_specplot", study, alleeg, *options, "threshold", 0.05)
                                _close_measure_plot(eeglab_backend, request)
                                _close_measure_plot(eeglab_backend, request)


def _factorial_study(*, n_channels: int = 6, n_components: int = 2) -> tuple[dict, list[dict]]:
    datasets = []
    srate = 64.0
    pnts = 64
    trials = 4
    seconds = np.arange(pnts, dtype=float) / srate
    for group_index, group in enumerate(("control", "patient")):
        for subject_index in range(2):
            subject = f"{group[0].upper()}{subject_index + 1:02d}"
            subject_shift = 0.025 * (subject_index + 1)
            for condition_index, condition in enumerate(("standard", "target")):
                condition_shift = condition_index * (0.35 + 0.04 * subject_index)
                amplitude = 1.0 + 0.2 * group_index + 0.1 * condition_index + subject_shift
                data = np.empty((n_channels, pnts, trials), dtype=float)
                activations = np.empty((n_components, pnts, trials), dtype=float)
                for trial in range(trials):
                    phase = trial * np.pi / 12
                    for channel in range(n_channels):
                        data[channel, :, trial] = (
                            amplitude * np.sin(2 * np.pi * (6 + channel) * seconds + phase)
                            + condition_shift
                            + 0.15 * group_index
                        )
                    for component in range(n_components):
                        activations[component, :, trial] = (amplitude + 0.1 * component) * np.sin(
                            2 * np.pi * (8 + 2 * component) * seconds + phase
                        ) + condition_shift
                mixing = np.zeros((n_channels, n_components), dtype=float)
                mixing[:n_components] = np.eye(n_components)
                if group_index == 1 and subject_index == 1 and condition_index == 1:
                    mixing[:, 0] *= -1
                weights = np.zeros((n_components, n_channels), dtype=float)
                weights[:, :n_components] = np.eye(n_components)
                chanlocs = []
                for channel in range(n_channels):
                    angle = 2 * np.pi * channel / n_channels
                    chanlocs.append(
                        {
                            "labels": f"Ch{channel + 1}",
                            "theta": float(np.degrees(angle)),
                            "radius": 0.4,
                            "X": float(np.cos(angle)),
                            "Y": float(np.sin(angle)),
                            "Z": 0.0,
                        }
                    )
                datasets.append(
                    {
                        "setname": f"{subject}_{condition}",
                        "subject": subject,
                        "condition": condition,
                        "group": group,
                        "session": 1,
                        "run": 1,
                        "data": data,
                        "nbchan": n_channels,
                        "pnts": pnts,
                        "trials": trials,
                        "srate": srate,
                        "xmin": 0.0,
                        "xmax": float(seconds[-1]),
                        "times": seconds * 1000.0,
                        "chanlocs": chanlocs,
                        "icaact": activations,
                        "icawinv": mixing,
                        "icaweights": weights,
                        "icasphere": np.eye(n_channels),
                        "icachansind": list(range(n_channels)),
                        "event": [],
                        "urevent": [],
                        "epoch": [{} for _trial in range(trials)],
                        "etc": {},
                    }
                )
    return pop_study(None, datasets, name="Deterministic 2 x 2 study")


def _component_clusters(study: dict) -> dict:
    study = deepcopy(study)
    parent = study["cluster"][0]
    dataset_ids = list(range(1, len(study["datasetinfo"]) + 1))
    study["cluster"] = [
        parent,
        {"name": "Cluster 1", "sets": [dataset_ids], "comps": [1] * len(dataset_ids), "child": []},
        {"name": "Cluster 2", "sets": [dataset_ids], "comps": [2] * len(dataset_ids), "child": []},
    ]
    parent["child"] = ["Cluster 1", "Cluster 2"]
    return study


def test_std_stat_fdr_preserves_undefined_samples_and_graded_thresholds():
    condition_a = np.asarray([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    condition_b = np.asarray([[1.0, 1.0, 1.0], [2.0, 4.0, 8.0]])

    pcond, _pgroup, _pinter = std_stat(
        [condition_a, condition_b],
        condstats="on",
        paired=["on", "off"],
        method="param",
        mcorrect="fdr",
    )
    assert np.isnan(pcond[0][0])
    assert np.isfinite(pcond[0][1])

    pvalue = float(pcond[0][1])
    masks, _group_masks, _interaction_masks = std_stat(
        [condition_a, condition_b],
        condstats="on",
        paired=["on", "off"],
        method="param",
        mcorrect="fdr",
        threshold=[pvalue / 2, min(pvalue * 2, 1.0)],
    )
    np.testing.assert_array_equal(masks[0], [0.0, 1.0])


def test_std_erpplot_groups_design_cells_and_returns_statistics_and_masks():
    study, alleeg = _factorial_study()
    study, alleeg = std_precomp(study, alleeg, [1], erp="on", recompute="on")
    study = pop_statparams(study, condstats="on", groupstats="on", threshold=np.nan, method="param")

    result = std_erpplot(study, alleeg, channels=[1], return_stats=True, plotstderr="on")
    _study, cells, times, pgroup, pcond, pinter, figure = result

    assert [[cell.shape for cell in row] for row in cells] == [[(64, 2), (64, 2)], [(64, 2), (64, 2)]]
    expected = ttest_rel(cells[0][0], cells[1][0], axis=-1).pvalue
    np.testing.assert_allclose(pcond[0], expected)
    assert len(pgroup) == 2
    assert len(pinter) == 3
    assert len(figure.axes) == 4
    assert figure.eegprep_plot_metadata["statistics"].condmask == []
    plt.close(figure)

    mask_result = std_erpplot(study, alleeg, channels=[1], threshold=0.05, return_stats=True)
    _study, _cells, _times, _pgroup, condition_masks, _pinter, mask_figure = mask_result
    assert set(np.unique(condition_masks[0])) <= {0.0, 1.0}
    np.testing.assert_array_equal(condition_masks, mask_figure.eegprep_plot_metadata["statistics"].condmask)
    plt.close(mask_figure)

    _study, _cells, _times, together = std_erpplot(
        study,
        alleeg,
        channels=[1],
        condstats="off",
        groupstats="off",
        plotconditions="together",
        plotgroups="together",
        plotsubjects="on",
    )
    assert len(together.axes) == 1
    assert len(together.axes[0].lines) == 12
    plt.close(together)


def test_std_erspplot_supports_clusters_subject_panels_and_channel_topographies():
    study, alleeg = _factorial_study()
    tf_params = {"cycles": 0, "nfreqs": 5, "timesout": 5, "baseline": np.nan}
    study, alleeg = std_precomp(study, alleeg, "channels", ersp="on", savetrials="on", erspparams=tf_params)
    study, alleeg = std_precomp(study, alleeg, "components", ersp="on", savetrials="on", erspparams=tf_params)
    study = _component_clusters(study)

    _study, cluster_cells, times, freqs, cluster_figure = std_erspplot(study, alleeg, clusters=[2, 3])
    assert cluster_cells[0][0].shape == (freqs.size, times.size, 2)
    assert len(cluster_figure.axes) >= 2
    plt.close(cluster_figure)

    _study, subject_cells, _times, _freqs, subject_figure = std_erspplot(
        study, alleeg, channels=[1], subject="C01", plotsubjects="on"
    )
    assert sum(cell.shape[-1] for row in subject_cells for cell in row) == 2
    assert len(subject_figure.axes) >= 4
    plt.close(subject_figure)

    _study, topo_cells, _times, _freqs, topo_figure = std_erspplot(
        study, alleeg, channels="channels", topofreq=8, topotime=400, caxis=[-3, 3]
    )
    assert topo_cells[0][0].shape[-2] == 6
    assert len(topo_figure.axes) == 4
    assert all(axis.images or not axis.get_visible() for axis in topo_figure.axes)
    plt.close(topo_figure)


def test_std_erspplot_channel_saved_trials_reproduce_the_cached_ersp():
    study, alleeg = _factorial_study()
    params = {"cycles": 0, "nfreqs": 5, "timesout": 5, "baseline": np.nan}
    study, alleeg = std_precomp(study, alleeg, [1], ersp="on", savetrials="on", recompute="on", erspparams=params)
    cache = study["changrp"][0]

    for dataset_index, trials in enumerate(cache["erspdatatrials"]):
        reconstructed = 10 * np.log10(np.mean(np.asarray(trials), axis=-1))
        np.testing.assert_allclose(reconstructed, np.asarray(cache["erspdata"])[dataset_index], atol=1e-12)
    _study, cells, times, freqs, figure = std_erspplot(study, alleeg, channels=[1])
    assert cells[0][0].shape == (freqs.size, times.size, 2)
    assert cache["measureinfo"]["trial_cache"]["erspdatatrials"] == "linear baseline-corrected power"
    plt.close(figure)


def test_std_erspplot_component_saved_trials_reproduce_the_cached_ersp():
    study, alleeg = _factorial_study()
    for info in study["datasetinfo"]:
        info["comps"] = [2]
    params = {"cycles": 0, "nfreqs": 5, "timesout": 5, "baseline": np.nan}
    study, alleeg = std_precomp(
        study, alleeg, "components", ersp="on", savetrials="on", recompute="on", erspparams=params
    )
    cache = study["cluster"][0]

    for dataset_index, component_trials in enumerate(cache["erspdatatrials"]):
        reconstructed = 10 * np.log10(np.mean(np.asarray(component_trials[0]), axis=-1))
        np.testing.assert_allclose(reconstructed, np.asarray(cache["erspdata"])[dataset_index, 0], atol=1e-12)
    _study, cells, times, freqs, figure = std_erspplot(study, alleeg, clusters=1, components=[2])
    assert cells[0][0].shape == (freqs.size, times.size, 2)
    plt.close(figure)


def test_std_itcplot_supports_centroids_component_panels_channels_and_subjects():
    study, alleeg = _factorial_study()
    tf_params = {"cycles": 0, "nfreqs": 4, "timesout": 4, "baseline": np.nan}
    study, alleeg = std_precomp(study, alleeg, [1], itc="on", savetrials="on", erspparams=tf_params)
    channel_cache = study["changrp"][0]
    for dataset_index, phases in enumerate(channel_cache["itcdatatrials"]):
        reconstructed = np.abs(np.mean(np.exp(1j * np.asarray(phases)), axis=-1))
        np.testing.assert_allclose(reconstructed, np.asarray(channel_cache["itcdata"])[dataset_index])
    study, alleeg = std_precomp(study, alleeg, "components", itc="on", erspparams=tf_params)
    study = _component_clusters(study)

    _study, cells, times, freqs, centroid = std_itcplot(study, alleeg, clusters=2, mode="centroid")
    assert cells[0][0].shape == (freqs.size, times.size, 2)
    assert np.nanmin(cells[0][0]) >= 0
    plt.close(centroid)

    _study, _cells, _times, _freqs, components = std_itcplot(study, alleeg, clusters=2, mode="comps")
    assert len(components.axes) >= 8
    plt.close(components)

    _study, channel_cells, _times, _freqs, channel_figure = std_itcplot(
        study, alleeg, channels=[1], subject="P01", plotsubjects="on"
    )
    assert sum(cell.shape[-1] for row in channel_cells for cell in row) == 2
    plt.close(channel_figure)


def test_std_specplot_supports_clusters_fdr_subject_traces_and_channel_topography():
    study, alleeg = _factorial_study()
    study, alleeg = std_precomp(study, alleeg, "channels", spec="on", recompute="on")
    study, alleeg = std_precomp(study, alleeg, "components", spec="on", recompute="on")
    study = _component_clusters(study)

    result = std_specplot(
        study,
        alleeg,
        clusters=2,
        condstats="on",
        plotconditions="together",
        threshold=0.05,
        mcorrect="fdr",
        return_stats=True,
    )
    _study, cells, frequencies, _pgroup, pcond, _pinter, figure = result
    assert cells[0][0].shape == (frequencies.size, 2)
    assert len(pcond) == 2
    assert figure.eegprep_plot_metadata["statistics"].mcorrect == "fdr"
    plt.close(figure)

    _study, _cells, _frequencies, subject_figure = std_specplot(
        study, alleeg, channels=[1], subject="C01", plotsubjects="on", plotconditions="together"
    )
    assert sum(len(axis.lines) for axis in subject_figure.axes) >= 4
    plt.close(subject_figure)

    _study, topo_cells, _frequencies, topo_figure = std_specplot(study, alleeg, channels="channels", topofreq=8)
    assert topo_cells[0][0].shape[-2] == 6
    assert len(topo_figure.axes) == 4
    plt.close(topo_figure)


def test_std_specplot_group_and_condition_layout_controls_preserve_design_cells():
    study, alleeg = _factorial_study()
    study, alleeg = std_precomp(study, alleeg, [1], spec="on", recompute="on")

    for plotconditions, plotgroups, expected_axes in (
        ("apart", "apart", 4),
        ("together", "apart", 2),
        ("apart", "together", 2),
        ("together", "together", 1),
    ):
        _study, cells, frequencies, figure = std_specplot(
            study,
            alleeg,
            channels=[1],
            plotconditions=plotconditions,
            plotgroups=plotgroups,
            plotsubjects="on",
        )
        assert [[cell.shape for cell in row] for row in cells] == [
            [(frequencies.size, 2), (frequencies.size, 2)],
            [(frequencies.size, 2), (frequencies.size, 2)],
        ]
        assert len(figure.axes) == expected_axes
        plt.close(figure)


@_reference("std_topoplot", "test_test_std_topoplot")
def test_reference_std_topoplot(eeglab_backend, eeglab_sample_study):
    study, alleeg = eeglab_sample_study
    for options in (
        {"clusters": "all", "mode": "centroid"},
        {"clusters": 3.0, "mode": "centroid"},
        {"clusters": 3.0, "mode": "comps"},
        {"clusters": 3.0, "comps": 4.0},
    ):
        eeglab_backend("std_topoplot", study, alleeg, **options, nargout=0)
        eeglab_backend("close", nargout=0)


def test_std_topoplot_draws_all_centroids_component_maps_and_selected_members():
    study, alleeg = _factorial_study()
    study, alleeg = std_precomp(study, alleeg, "components", erp="on", scalp="on", recompute="on")
    study = _component_clusters(study)

    study, all_figure = std_topoplot(study, alleeg, clusters="all", mode="centroid")
    assert len(all_figure.axes) == 2
    assert all(cluster.get("topo") for cluster in study["cluster"][1:])
    plt.close(all_figure)

    study, component_figure = std_topoplot(study, alleeg, clusters=2, mode="comps")
    assert len(component_figure.axes) == len(study["cluster"][1]["comps"]) + 1
    assert set(study["cluster"][1]["topopol"]) <= {-1, 1}
    plt.close(component_figure)

    _study, selected_figure = std_topoplot(study, alleeg, clusters=2, components=[4], mode="comps")
    assert len(selected_figure.axes) == 2
    assert selected_figure.axes[1].get_title().endswith("/IC1")
    plt.close(selected_figure)
