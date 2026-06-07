from __future__ import annotations

import ast
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pytest

import eegprep
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.std_pac import std_pac
from eegprep.functions.studyfunc.std_pacplot import std_pacplot
from eegprep.functions.studyfunc.std_readpac import std_readpac
from eegprep.functions.timefreqfunc._pac_support import PAC_UNSUPPORTED_MESSAGE
from eegprep.functions.timefreqfunc.pac import PacResult, pac
from eegprep.functions.timefreqfunc.pac_cont import PacContResult, pac_cont
from tests.fixtures import create_test_eeg


def test_pac_computes_frequency_pair_grid_for_epoched_data():
    amp, phase, srate = _coupled_trials()

    result = pac(amp, phase, srate, freqs=[20, 40], freqs2=[4, 8], nfreqs=4, ntimesout=8)
    corr = pac(amp, phase, srate, freqs=[20, 40], freqs2=[4, 8], nfreqs=4, ntimesout=8, method="corrsin")

    assert isinstance(result, PacResult)
    assert result.pac.shape == (4, 2, 8)
    assert result.times.size == 8
    assert np.iscomplexobj(result.pac)
    assert np.isfinite(np.abs(result.pac)).all()
    assert corr.pac.shape == result.pac.shape
    assert not np.iscomplexobj(corr.pac)
    assert eegprep.pac is pac


def test_pac_cont_computes_sliding_window_modulation_and_pvalues():
    amp, phase, srate = _coupled_trials()

    result = pac_cont(
        amp[:, 0],
        phase[:, 0],
        srate,
        freqphase=[4, 8],
        freqamp=[20, 40],
        winsize=32,
        ntimesout=6,
        alpha=0.1,
        baseline=[0, 1000],
        nofig="on",
    )

    assert isinstance(result, PacContResult)
    assert result.pac.shape == result.times.shape == result.pvalues.shape
    assert result.indices.min() >= 1
    assert np.all(result.pac >= 0)
    assert np.all((result.pvalues >= 0) & (result.pvalues <= 1))
    assert eegprep.pac_cont is pac_cont


def test_pac_reports_explicit_unsupported_statistics_and_latphase():
    amp, phase, srate = _coupled_trials()

    with pytest.raises(NotImplementedError, match="PAC bootstrap significance"):
        pac(amp, phase, srate, alpha=0.05)
    with pytest.raises(NotImplementedError, match="PAC bootstrap significance"):
        pac(amp, phase, srate, method="latphase")

    assert "not silently emulated" in PAC_UNSUPPORTED_MESSAGE


def test_std_pac_computes_study_cache_and_std_pacplot_reads_it():
    study, alleeg = _study_pair()

    pacdata, times, freqs, params, dataset_command = std_pac(
        alleeg[0],
        channels1=[1],
        channels2=[1],
        freqs=[20, 40],
        freqphase=[6],
        nfreqs=4,
        ntimesout=8,
        return_com=True,
    )
    study, alleeg, study_command = std_pac(
        study,
        alleeg,
        channels1=[1],
        channels2=[1],
        freqs=[20, 40],
        freqphase=[6],
        nfreqs=4,
        ntimesout=8,
        return_com=True,
    )
    _study, read_data, read_times, read_freqs = std_readpac(study, alleeg, channels1=[1], channels2=[1])
    _study, plot_data, plot_times, plot_freqs, figure, plot_command = std_pacplot(
        study, alleeg, channels1=[1], return_com=True
    )

    assert pacdata.shape == (1, 1, freqs.size, times.size)
    assert params["freqphase"] == [6.0]
    assert np.asarray(study["changrp"][0]["pacdata"]).shape == (2, freqs.size, times.size)
    np.testing.assert_allclose(read_data[0], study["changrp"][0]["pacdata"])
    np.testing.assert_allclose(plot_data[0], read_data[0])
    np.testing.assert_allclose(read_times, plot_times)
    np.testing.assert_allclose(read_freqs, plot_freqs)
    assert study["changrp"][0]["measureinfo"]["pac"]["channels2"] == [1]
    assert dataset_command.startswith("PACDATA, PACTIMES, PACFREQS, PARAMETERS = std_pac(")
    assert study_command.startswith("STUDY, ALLEEG = std_pac(")
    assert plot_command.startswith("STUDY, PACDATA, PACTIMES, PACFREQS, FIGURE = std_pacplot(")
    ast.parse(dataset_command)
    ast.parse(study_command)
    ast.parse(plot_command)
    assert eegprep.std_pac is std_pac
    assert eegprep.std_pacplot is std_pacplot
    assert eegprep.std_readpac is std_readpac
    plt.close(figure)


def test_std_readpac_reads_and_slices_eegprep_owned_channel_cache():
    study, alleeg = _study_pair()
    raw = np.arange(2 * 4 * 5, dtype=float).reshape(2, 4, 5)
    study["changrp"] = [
        {
            "name": "Ch1",
            "channels": ["Ch1"],
            "inds": [1],
            "pacdata": raw,
            "pacfreqs": [4.0, 8.0, 12.0, 16.0],
            "pactimes": [-100.0, 0.0, 100.0, 200.0, 300.0],
        }
    ]

    _study, pacdata, pactimes, pacfreqs = std_readpac(
        study,
        alleeg,
        channels=[1],
        timerange=[0.0, 200.0],
        freqrange=[8.0, 16.0],
    )

    np.testing.assert_allclose(pacdata[0], raw[:, 1:4, 1:4])
    np.testing.assert_allclose(pactimes, [0.0, 100.0, 200.0])
    np.testing.assert_allclose(pacfreqs, [8.0, 12.0, 16.0])


def test_std_readpac_rejects_missing_or_malformed_pac_caches():
    study, alleeg = _study_pair()

    with pytest.raises(NotImplementedError, match="PAC reading requires EEGPrep-owned pacdata caches"):
        std_readpac(study, alleeg, clusters=1)

    malformed = deepcopy(study)
    malformed["cluster"][0]["pacdata"] = np.ones((2, 3, 4))
    malformed["cluster"][0]["pacfreqs"] = [4.0, 8.0]
    malformed["cluster"][0]["pactimes"] = [0.0, 100.0, 200.0, 300.0]
    with pytest.raises(ValueError, match="PAC cache shape"):
        std_readpac(malformed, alleeg, clusters=1)

    with pytest.raises(ValueError, match="Unknown std_readpac option"):
        std_readpac(study, alleeg, clusters=1, unsupported="on")


def _coupled_trials():
    srate = 128
    times = np.arange(128) / srate
    phase = np.column_stack(
        [
            np.sin(2 * np.pi * 6 * times),
            np.sin(2 * np.pi * 6 * times + 0.4),
            np.sin(2 * np.pi * 6 * times + 0.8),
        ]
    )
    amp = np.column_stack(
        [
            (1.0 + 0.5 * np.sin(2 * np.pi * 6 * times + offset)) * np.sin(2 * np.pi * 30 * times)
            for offset in (0.0, 0.4, 0.8)
        ]
    )
    return amp, phase, srate


def _study_pair():
    first = create_test_eeg(n_channels=2, n_samples=128, n_trials=3, srate=128)
    first.update({"setname": "one", "subject": "S01", "condition": "target"})
    second = deepcopy(first)
    second.update({"setname": "two", "subject": "S02", "condition": "standard"})
    second["data"] = second["data"] * 2.0
    return pop_study(None, [first, second], name="PAC cache study")
