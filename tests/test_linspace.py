"""MATLAB linspace spacing."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.miscfunc.linspace import linspace
from eegprep.functions.popfunc.plot_utils import eeg_times_ms
from eegprep.functions.sigprocfunc.envtopo import envtopo
from eegprep.functions.timefreqfunc._pac_support import _pac_cont_windows
from eegprep.functions.timefreqfunc.dftfilt2 import dftfilt2
from eegprep.functions.timefreqfunc.tf_cycle_calc import tf_cycle_calc
from eegprep.functions.timefreqfunc.timefreq import timefreq
from eegprep.plugins.firfilt.plotfresp import plotfresp


def test_linspace_matches_matlab_values():
    got = linspace(0.0, 1.0, 7)
    expected = np.array(
        [
            float.fromhex("0x0.0p+0"),
            float.fromhex("0x1.5555555555555p-3"),
            float.fromhex("0x1.5555555555555p-2"),
            float.fromhex("0x1.0000000000000p-1"),
            float.fromhex("0x1.5555555555555p-1"),
            float.fromhex("0x1.aaaaaaaaaaaabp-1"),
            float.fromhex("0x1.0000000000000p+0"),
        ]
    )
    np.testing.assert_array_equal(got, expected)
    assert got[5] != np.linspace(0.0, 1.0, 7)[5]
    np.testing.assert_array_equal(linspace(1.0, 2.0, 1), np.array([2.0]))


def test_linspace_keeps_matlab_half_sample_ties():
    start = float.fromhex("-0x1.b19b366cd9b36p+8")
    stop = float.fromhex("0x1.a99b366cd9b36p+8")
    got = linspace(start, stop, 111)
    numpy_grid = np.linspace(start, stop, 111)
    ties = {
        20: "-0x1.1562c58b162c5p+8",
        72: "0x1.0193264c99328p+7",
        77: "0x1.4faf5ebd7af5cp+7",
        79: "0x1.6eeddbb76edd8p+7",
        106: "0x1.8a5cb972e5cbap+8",
    }
    for index, hex_value in ties.items():
        assert got[index].hex() == hex_value
        assert got[index] != numpy_grid[index]
    assert got[0] == start
    assert got[-1] == stop


def test_eeg_times_match_matlab_linspace_for_seven_samples():
    eeg = eeg_checkset(
        {
            "data": np.zeros((1, 7)),
            "nbchan": 1,
            "pnts": 7,
            "trials": 1,
            "srate": 6000.0,
            "xmin": 0.0,
            "xmax": 0.001,
        }
    )
    expected = np.array(
        [
            float.fromhex("0x0.0p+0"),
            float.fromhex("0x1.5555555555555p-3"),
            float.fromhex("0x1.5555555555555p-2"),
            float.fromhex("0x1.0000000000000p-1"),
            float.fromhex("0x1.5555555555555p-1"),
            float.fromhex("0x1.aaaaaaaaaaaabp-1"),
            float.fromhex("0x1.0000000000000p+0"),
        ]
    )
    np.testing.assert_array_equal(eeg["times"], expected)
    assert eeg["times"][5] != np.linspace(0.0, 1.0, 7)[5]
    rebuilt = eeg_times_ms({"pnts": 7, "xmin": 0.0, "xmax": 0.001, "times": []})
    np.testing.assert_array_equal(rebuilt, expected)


def test_timefreq_frequency_grid_matches_matlab_linspace():
    result = timefreq(
        np.zeros((1, 128)),
        256,
        cycles=1,
        winsize=8,
        freqs=[3, 10],
        nfreqs=6,
        verbose="off",
    )
    assert result.freqs[3].hex() == "0x1.ccccccccccccdp+2"
    assert result.freqs[3] != np.linspace(3.0, 10.0, 6)[3]

    single = timefreq(
        np.zeros((1, 128)),
        256,
        cycles=1,
        winsize=8,
        freqs=[3, 10],
        nfreqs=1,
        verbose="off",
    )
    np.testing.assert_array_equal(single.freqs, np.array([10.0]))


def test_cycle_grids_match_matlab_linspace_and_logspace():
    freqs = np.arange(2.0, 8.0)
    linear = tf_cycle_calc(freqs, [1.0, 5.0], "cycles", plot=False)
    assert linear.cycles[3].hex() == "0x1.b333333333333p+1"
    assert linear.cycles[3] != np.linspace(1.0, 5.0, freqs.size)[3]

    logarithmic = tf_cycle_calc(np.arange(2.0, 9.0), [3.0, 8.0], "cycles", log_spaced=True, plot=False)
    assert logarithmic.cycles[5].hex() == "0x1.b2c91e6a9e53fp+2"
    assert logarithmic.cycles[5] != np.logspace(np.log10(3.0), np.log10(8.0), 7)[5]


def test_one_frequency_cycle_range_uses_the_matlab_stop_value():
    wavelets = dftfilt2([10.0], [3.0, 8.0], 100.0)
    assert wavelets[0].size == 81


def test_plotfresp_frequency_grid_matches_matlab_linspace():
    _figure, _axes, response = plotfresp([1.0], 1, nfft=10, fs=128, show=False)
    try:
        assert response["frequency_hz"][3].hex() == "0x1.3333333333333p+5"
        assert response["frequency_hz"][3] != np.linspace(0.0, 64.0, response["frequency_hz"].size)[3]
    finally:
        plt.close("all")


def test_envtopo_times_follow_the_matlab_colon():
    frames = 111
    activation = np.zeros((1, frames))
    activation[0, 55] = 1.0
    result = envtopo(activation, np.array([[1.0]]), timerange=[-500.0, 492.0], compsplot=1)
    try:
        assert int(result.compframes[0]) == 55
        assert float(result.comptimes[0]).hex() == "-0x1.0000000000004p+2"
        assert result.comptimes[0] != np.linspace(-500.0, 492.0, frames)[55]
    finally:
        plt.close(result.figure)


def test_single_pac_window_selects_the_matlab_linspace_stop():
    _times, indices = _pac_cont_windows(20, np.array([0.0, 100.0]), 4, None, 1)
    np.testing.assert_array_equal(indices, np.array([18]))
