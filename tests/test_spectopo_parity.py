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

import numpy as np
import scipy.io

from eegprep import pop_loadset, pop_saveset
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.sigprocfunc.spectopo import spectopo

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
        self.EEG = pop_loadset(os.path.join(local_url, "eeglab_data.set"))

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


if __name__ == "__main__":
    unittest.main()
