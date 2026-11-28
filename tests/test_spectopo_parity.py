"""Numerical parity between EEGPrep and MATLAB/EEGLAB ``spectopo``.

Runs EEGLAB's ``spectopo`` through the MATLAB engine and compares the returned
channel log-power spectra (dB) against EEGPrep's ``spectopo`` on the identical
dataset. Requires the MATLAB engine plus an EEGLAB checkout (via
``eeglabcompat.get_eeglab``) and is skipped when either is unavailable, e.g. in
CI. No MATLAB output is committed; the reference is regenerated live, mirroring
``test_eeg_rpsd_parity``.
"""

# Force single-threaded BLAS for deterministic numerics (mirrors
# test_eeg_rpsd_parity); must be set before numpy imports.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.io

from eegprep import pop_loadset, pop_saveset
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.pop_spectopo import pop_spectopo
from eegprep.functions.sigprocfunc.spectopo import spectopo
from tests.fixtures import create_test_eeg_with_ica

local_url = os.path.join(os.path.dirname(__file__), "../sample_data/")

# Absolute dB tolerance. Observed max |Δ| < 1e-5 dB vs MATLAB pwelch on the sample
# data; 1e-2 guards against real regressions (miscalibration) without float flakiness.
SPECTRA_ATOL_DB = 1e-2


class TestSpectopoParity(unittest.TestCase):
    """Parity between Python and MATLAB spectopo channel spectra."""

    def setUp(self):
        try:
            self.eeglab = get_eeglab("MAT", auto_file_roundtrip=False)
        except Exception as e:
            self.skipTest(f"MATLAB/EEGLAB not available: {e}")
        # Epoched dataset exercises the per-epoch averaging path; the continuous
        # (single-epoch) path is trivially the same as ``pwelch`` and is not the
        # regression this test guards.
        self.EEG = pop_loadset(os.path.join(local_url, "eeglab_data_epochs_ica.set"))

    def test_channel_spectra_match_matlab(self):
        """EEGPrep spectopo channel spectra match EEGLAB spectopo (pwelch), in dB."""
        # Python spectra (dB), no plotting.
        py_spectra, py_freqs = spectopo(self.EEG["data"], self.EEG["pnts"], float(self.EEG["srate"]), plot="off")[:2]
        py_freqs = np.asarray(py_freqs, dtype=float).ravel()

        # MATLAB spectra on the identical dataset via a .set roundtrip.
        temp_file = tempfile.mktemp(suffix=".set")
        pop_saveset(self.EEG, temp_file)
        matlab_code = f"""
        set(0, 'DefaultFigureVisible', 'off');
        EEG = pop_loadset('{temp_file}');
        [spectra, freqs] = spectopo(EEG.data, EEG.pnts, EEG.srate);
        close all; set(0, 'DefaultFigureVisible', 'on');
        save('{temp_file}.mat', 'spectra', 'freqs');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        mat_data = scipy.io.loadmat(temp_file + ".mat")
        ml_spectra = np.asarray(mat_data["spectra"], dtype=float)
        ml_freqs = np.asarray(mat_data["freqs"], dtype=float).ravel()

        # Clean up temp files.
        os.remove(temp_file)
        os.remove(temp_file + ".mat")
        if os.path.exists(temp_file.replace(".set", ".fdt")):
            os.remove(temp_file.replace(".set", ".fdt"))

        self.assertEqual(py_spectra.shape, ml_spectra.shape)
        np.testing.assert_allclose(py_freqs, ml_freqs, rtol=0, atol=1e-9, err_msg="frequency grids differ")
        np.testing.assert_allclose(
            py_spectra,
            ml_spectra,
            rtol=0,
            atol=SPECTRA_ATOL_DB,
            err_msg="EEGPrep spectopo channel spectra differ from EEGLAB (pwelch) beyond tolerance",
        )


def test_pop_spectopo_splits_spectopo_options_before_topoplot():
    """spectopo-level options (limits, reref, mapnorm, rmdc, plotmean, title) must not leak into topoplot."""
    eeg = create_test_eeg_with_ica(n_channels=6, n_samples=128, n_trials=2, n_components=4)
    eeg["srate"] = 128.0
    eeg["xmin"] = 0.0
    eeg["xmax"] = 127.0 / 128.0
    eeg["times"] = np.linspace(0.0, 1000.0 * 127.0 / 128.0, 128)

    res = pop_spectopo(
        eeg,
        1,
        [0, 1000],
        "EEG",
        "freq",
        [10],
        "freqrange",
        [2, 25],
        "limits",
        [2, 25, -40, 20, -5, 5],
        "reref",
        "averef",
        "rmdc",
        "on",
        "plotmean",
        "on",
        "mapnorm",
        np.ones(6),
        "title",
        "Channel spectrum",
        plot="off",
    )
    try:
        spec_ax = res["figure"].axes[0]
        assert spec_ax.get_xlim() == pytest.approx((2.0, 25.0))
        assert spec_ax.get_ylim() == pytest.approx((-40.0, 20.0))
        # plotmean='on' draws 1 mean trace plus 1 vertical marker line at 10 Hz.
        assert len(spec_ax.lines) == 2
        # Scalp map axis receives cmin/cmax from limits[4:6] (with EEGLAB's 5% clim margin).
        topo_ax = next(ax for ax in res["figure"].axes[1:] if ax.images)
        np.testing.assert_allclose(topo_ax.images[0].get_clim(), [-5.25, 5.25])
    finally:
        plt.close(res["figure"])

    # Unimplemented EEGLAB spectopo option names pop_spectopo rather than leaking into topoplot.
    with pytest.raises(ValueError, match=r"pop_spectopo: 'boundaries' is an EEGLAB spectopo option"):
        pop_spectopo(eeg, 1, [0, 1000], "EEG", "freq", [10], "boundaries", [50], plot="off")

    # Truly unknown option reaches topoplot and raises topoplot's unknown-parameter error.
    with pytest.raises(ValueError, match=r"topoplot: Unknown input parameter 'foobar'"):
        pop_spectopo(eeg, 1, [0, 1000], "EEG", "freq", [10], "foobar", 1, plot="off")


def test_pop_spectopo_plotchans_empty_defaults_and_freqfac():
    """plotchans subsets channel spectra, empty [] numeric defaults work, and freqfac scales both window types."""
    eeg = create_test_eeg_with_ica(n_channels=6, n_samples=128, n_trials=2, n_components=4)
    eeg["srate"] = 128.0
    eeg["xmin"] = 0.0
    eeg["xmax"] = 127.0 / 128.0
    eeg["times"] = np.linspace(0.0, 1000.0 * 127.0 / 128.0, 128)

    # plotchans in channel mode slices spectra & traces (spectopo.m:263-268) without leaking to topoplot.
    subset = pop_spectopo(eeg, 1, [0, 1000], "EEG", "freq", [10], "plotchans", [1, 3, 5], plot="off")
    try:
        assert subset["spectra"].shape[0] == 3
        # 3 channel traces + 1 vertical marker at 10 Hz.
        assert len(subset["figure"].axes[0].lines) == 4
    finally:
        plt.close(subset["figure"])

    # plotchans in component mode is ignored (spectopo.m:263) and never leaks to topoplot.
    comp = pop_spectopo(eeg, 0, [0, 1000], "EEG", "freq", [10], "plotchans", [1, 2, 3], "nicamaps", 2, plot="off")
    try:
        assert comp["spectra"].shape[0] == 4
    finally:
        plt.close(comp["figure"])

    with pytest.raises(ValueError, match=r"spectopo: plotchans must be 1-based indices"):
        pop_spectopo(eeg, 1, [0, 1000], "EEG", "plotchans", [0, 99], plot="off")

    # EEGLAB empty [] defaults for numeric options fall back cleanly without IndexError.
    empty_defaults = pop_spectopo(
        eeg,
        1,
        [0, 1000],
        "EEG",
        "winsize",
        [],
        "nfft",
        [],
        "overlap",
        [],
        "blckhn",
        [],
        "freqfac",
        [],
        "mapnorm",
        [],
        plot="off",
    )
    try:
        assert empty_defaults["spectra"].shape == (6, 65)
    finally:
        plt.close(empty_defaults["figure"])

    # freqfac=2 doubles FFT grid resolution for both hamming and blackmanharris (spectopo.m:910-917).
    ham1 = pop_spectopo(eeg, 1, [0, 1000], "EEG", "wintype", "hamming", "freqfac", 1, plot="off")
    ham2 = pop_spectopo(eeg, 1, [0, 1000], "EEG", "wintype", "hamming", "freqfac", 2, plot="off")
    bh1 = pop_spectopo(eeg, 1, [0, 1000], "EEG", "wintype", "blackmanharris", "freqfac", 1, plot="off")
    bh2 = pop_spectopo(eeg, 1, [0, 1000], "EEG", "wintype", "blackmanharris", "freqfac", 2, plot="off")
    try:
        assert ham1["freqs"].size == 65
        assert ham2["freqs"].size == 129
        assert bh1["freqs"].size == 33
        assert bh2["freqs"].size == 65
    finally:
        for item in (ham1, ham2, bh1, bh2):
            plt.close(item["figure"])

    with pytest.raises(ValueError, match="freqfac must be a positive integer"):
        pop_spectopo(eeg, 1, [0, 1000], "EEG", "freqfac", 0, plot="off")


if __name__ == "__main__":
    unittest.main()
