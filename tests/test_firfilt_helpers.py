from __future__ import annotations

import os
import unittest

import matplotlib
import numpy as np
import pytest

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.plugins.firfilt.fir_filterdcpadded import fir_filterdcpadded
from eegprep.plugins.firfilt.findboundaries import findboundaries
from eegprep.plugins.firfilt.firfiltreport import firfiltreport
from eegprep.plugins.firfilt.firfiltsplit import firfiltsplit
from eegprep.plugins.firfilt.firgauss import firgauss
from eegprep.plugins.firfilt.invfirwsord import invfirwsord
from eegprep.plugins.firfilt.invkaiserbeta import invkaiserbeta
from eegprep.plugins.firfilt.kaiserbeta import kaiserbeta
from eegprep.plugins.firfilt.minphaserceps import minphaserceps
from eegprep.plugins.firfilt.plotfresp import plotfresp
from eegprep.plugins.firfilt.windows import windows

try:
    from .fixtures import create_test_eeg
except (ImportError, ValueError):
    from fixtures import create_test_eeg


matplotlib.use("Agg", force=True)


def test_kaiserbeta_matches_eeglab_formula_and_inverse_roundtrips():
    beta = kaiserbeta(0.001)

    assert beta == pytest.approx(5.65326, abs=1e-10)
    assert invkaiserbeta(beta) == pytest.approx(0.001, rel=1e-10)
    assert invkaiserbeta(0) == pytest.approx(10 ** (-21 / 20))


def test_firws_and_firwsord_are_owned_by_firfilt_plugin():
    """The FIR design lives in the firfilt plugin, not clean_rawdata's private helper.

    Locks the ownership move: ``firfilt/firws.py`` and ``firfilt/firwsord.py`` now
    define the canonical implementations, so the design helpers must no longer leak
    out of ``clean_rawdata.private.sigproc`` and importers resolve into firfilt.
    """
    from eegprep.plugins.clean_rawdata.private import sigproc
    from eegprep.plugins.firfilt.firws import firws
    from eegprep.plugins.firfilt.firwsord import firwsord

    assert firws.__module__ == "eegprep.plugins.firfilt.firws"
    assert firwsord.__module__ == "eegprep.plugins.firfilt.firwsord"
    assert not hasattr(sigproc, "firws")
    assert not hasattr(sigproc, "firwsord")

    # The firwsord order still feeds firws to produce a valid type-I linear-phase kernel.
    fs, cutoff, df = 500.0, 0.5, 1.0
    m, _dev = firwsord("hamming", fs, df)
    b, a = firws(m, cutoff / (fs / 2.0), "high")
    assert a == 1.0
    assert b.size == m + 1
    np.testing.assert_allclose(b, b[::-1], atol=1e-12)


def test_clean_rawdata_fir_design_helpers_are_owned_by_firfilt_plugin():
    """Clean rawdata imports FIR design helpers downward from firfilt."""
    from eegprep.plugins.clean_rawdata.private import sigproc
    from eegprep.plugins.firfilt.design import design_fir, design_kaiser

    assert design_fir.__module__ == "eegprep.plugins.firfilt.design"
    assert design_kaiser.__module__ == "eegprep.plugins.firfilt.design"
    assert not hasattr(sigproc, "design_fir")
    assert not hasattr(sigproc, "design_kaiser")

    window = design_kaiser(0.06, 0.08, 75.0, True)
    coeffs = design_fir(234, [0.0, 0.06, 0.08, 1.0], [0, 0, 1, 1], w=window)
    assert window.size % 2 == 1
    assert coeffs.size == 235


def test_invfirwsord_returns_transition_width_and_window_deviation():
    df, dev = invfirwsord("hamming", 500, 826)

    assert df == pytest.approx(3.3 / 826 * 500)
    assert dev == pytest.approx(0.0022)

    kaiser_df, kaiser_dev = invfirwsord("kaiser", 500, 1800, 0.001)
    assert kaiser_df == pytest.approx((60 - 8) / (2.285 * 2 * np.pi * (1800 - 1)) * 500)
    assert kaiser_dev == pytest.approx(0.001)


def test_windows_match_eeglab_symmetric_shapes():
    np.testing.assert_allclose(windows("hann", 5), [0.0, 0.5, 1.0, 0.5, 0.0], atol=1e-15)
    np.testing.assert_allclose(windows("hamming", 4), [0.08, 0.77, 0.77, 0.08], atol=1e-15)
    np.testing.assert_allclose(windows("tukey", 6, 0.5), [0.0, 0.9045085, 1.0, 1.0, 0.9045085, 0.0])


def test_minphaserceps_keeps_length_and_reduces_energy_delay():
    b = np.r_[0.0, 0.1, 0.25, 0.3, 0.25, 0.1, 0.0]

    out = minphaserceps(b, upsampling_factor=128)

    assert out.shape == b.shape
    original_centroid = np.sum(np.arange(b.size) * b**2) / np.sum(b**2)
    new_centroid = np.sum(np.arange(out.size) * out**2) / np.sum(out**2)
    assert new_centroid < original_centroid


def test_firgauss_matches_closed_form_lowpass_coefficients():
    b = firgauss(25, 500)

    assert b.size % 2 == 1
    assert b.size == 25
    assert np.argmax(b) == 12
    assert np.sum(b) == pytest.approx(0.9991776738, rel=1e-10)


def test_findboundaries_returns_eeglab_boundary_latencies():
    events = [
        {"type": "stim", "latency": 10},
        {"type": "boundary", "latency": 40.2},
        {"type": "boundary", "latency": 40.4},
        {"type": "boundary", "latency": 100.5},
    ]

    np.testing.assert_array_equal(findboundaries(events), [1, 40, 101])
    old = EEG_OPTIONS["option_boundary99"]
    try:
        EEG_OPTIONS["option_boundary99"] = 0
        np.testing.assert_array_equal(findboundaries([{"type": -99, "latency": 12.6}]), [1])
        EEG_OPTIONS["option_boundary99"] = 1
        np.testing.assert_array_equal(findboundaries([{"type": -99, "latency": 12.6}]), [1, 13])
    finally:
        EEG_OPTIONS["option_boundary99"] = old
    np.testing.assert_array_equal(findboundaries([]), [1])


def test_fir_filterdcpadded_matches_internal_segment_orientation():
    data = np.zeros((12, 2), dtype=float)
    data[4, 0] = 1
    data[7, 1] = 1
    b = np.ones(5, dtype=float) / 5

    filtered = fir_filterdcpadded(b, 1, data)

    assert filtered.shape == data.shape
    np.testing.assert_allclose(filtered[:, 0], [0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0])


def test_firfiltsplit_respects_boundary_events_and_channel_indices():
    eeg = create_test_eeg(n_channels=2, n_samples=80, srate=100.0, n_trials=1)
    eeg["data"] = np.zeros((2, 80), dtype=float)
    eeg["data"][0, 38] = 1.0
    eeg["data"][1, 38] = 1.0
    eeg["event"] = [{"type": "boundary", "latency": 40.5}]
    b = np.ones(5, dtype=float) / 5

    out = firfiltsplit(eeg, b, chaninds=[1])

    assert np.any(out["data"][0, :40])
    np.testing.assert_allclose(out["data"][0, 40:], 0, atol=1e-12)
    np.testing.assert_allclose(out["data"][1], eeg["data"][1])


def test_firfiltreport_formats_eeglab_style_summary():
    report = firfiltreport(
        func="pop_firws",
        family="hamming-windowed sinc FIR",
        type="lowpass",
        dir="onepass-zerophase",
        order=100,
        fs=250,
        fc=40,
        df=8.25,
        pbdev=0.0022,
        sbatt=0.0022,
    )

    assert "pop_firws() - lowpass filtering data: onepass-zerophase, order 100" in report
    assert "cutoff (-6 dB) 40 Hz" in report
    assert "transition width 8.2 Hz" in report
    assert "max. passband deviation 0.0022" in report


def test_plotfresp_returns_response_data_and_matplotlib_figure():
    b = np.ones(9, dtype=float) / 9

    fig, axes, response = plotfresp(b, nfft=64, fs=128, show=False)

    assert len(axes) == 5
    assert response["frequency_hz"].shape == (33,)
    assert response["magnitude_linear"][0] == pytest.approx(1.0)
    assert fig.axes[0].get_title() == "Impulse response"


@unittest.skipIf(os.getenv("EEGPREP_SKIP_MATLAB") == "1", "MATLAB not available")
class TestFirfiltHelperMatlabParity(unittest.TestCase):
    def setUp(self):
        try:
            self.eeglab = get_eeglab("MAT")
        except Exception as exc:
            self.skipTest(f"MATLAB not available: {exc}")

    def test_order_helper_parity(self):
        self.assertAlmostEqual(kaiserbeta(0.001), float(self.eeglab.kaiserbeta(0.001)), places=10)
        self.assertAlmostEqual(invkaiserbeta(5.65326), float(self.eeglab.invkaiserbeta(5.65326)), places=10)
        df, dev = invfirwsord("hamming", 500, 826)
        ml_df, ml_dev = self.eeglab.invfirwsord("hamming", 500.0, 826.0, nargout=2)
        self.assertAlmostEqual(df, float(ml_df), places=10)
        self.assertAlmostEqual(dev, float(ml_dev), places=10)

    def test_window_and_minphase_parity(self):
        np.testing.assert_allclose(
            windows("blackmanharris", 9), np.asarray(self.eeglab.windows("blackmanharris", 9.0)).ravel()
        )
        b = np.r_[0.0, 0.1, 0.25, 0.3, 0.25, 0.1, 0.0]
        np.testing.assert_allclose(minphaserceps(b), np.asarray(self.eeglab.minphaserceps(b)).ravel(), atol=1e-10)
