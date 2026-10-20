"""
Test parity between Python and MATLAB implementations of eeg_rpsd.

This test compares the Python implementation against the MATLAB/EEGLAB reference.
Multithreading is disabled for deterministic numerical results.
"""

# Disable multithreading for deterministic numerical results
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import unittest
import numpy as np
import tempfile
import scipy.io
from eegprep import pop_loadset, pop_saveset, eeg_rpsd
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab

local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')


class TestEegRpsdParity(unittest.TestCase):
    """Test parity between Python and MATLAB eeg_rpsd implementations."""

    def setUp(self):
        """Set up test fixtures."""
        # Try to get MATLAB engine
        try:
            self.eeglab = get_eeglab('MAT', auto_file_roundtrip=False)
            self.matlab_available = True
        except Exception as e:
            self.matlab_available = False
            self.skipTest(f"MATLAB not available: {e}")

        test_file = os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')
        self.EEG = pop_loadset(test_file)

    def test_parity_default_nfreqs(self):
        """Compare default spectra on the activations pop_loadset recomputes.

        ``pop_loadset`` fills ``icaact`` (``eeg_checkset.m:1030-1032``).
        ``eeg_rpsd`` reads that array and does not recompute it, so both
        sides window the loaded activations.
        """
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        psdmed = eeg_rpsd(EEG);
        icaact = EEG.icaact;
        save('{temp_file}.mat', 'psdmed', 'icaact');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        mat_data = scipy.io.loadmat(temp_file + '.mat', mat_dtype=True)
        ml_result = mat_data['psdmed']
        compared = self.EEG.copy()
        compared['icaact'] = mat_data['icaact']
        py_result = eeg_rpsd(compared)

        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        np.testing.assert_allclose(
            py_result,
            ml_result,
            rtol=2e-5,
            atol=1e-8,
            err_msg="eeg_rpsd results differ beyond tolerance (default nfreqs)",
        )

    def test_parity_double_activations(self):
        """Compare spectra for one shared float64 activation array.

        The activations ``pop_loadset`` recomputes are cast with MATLAB
        ``double`` and passed to both sides.

        Let ``n`` be the radix-2 FFT length ``min(pnts, srate)`` and
        ``u = eps(float64) / 2``. One bin rounds a window multiply (1 flop),
        ``log2(n)`` butterflies (8 real flops each: 4 multiplies and 2 adds
        for the twiddle, then 2 adds), ``|z|^2`` (3 flops),
        ``sum(window.^2)`` (``2n - 1`` flops), a multiply by the sampling
        rate, a division, and a multiply by that scale. Powers of two are
        exact, so the leading factor of 2 and the Nyquist halving add no
        rounding. An even segment count averages the two central median
        samples (1 add); an odd count only selects one. With ``k`` those
        roundings, ``γ_k = k u / (1 - k u)``. Each evaluation can miss the
        power by ``γ_k``, so the two powers can differ by ``2 γ_k``
        relatively. ``20*log10`` turns that relative error into an absolute
        decibel error of ``(20/ln(10)) * 2 γ_k``. ``rtol`` is ``2 γ_k`` and
        ``atol`` is that decibel figure.
        """
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        n_points = min(int(self.EEG['pnts']), int(self.EEG['srate']))
        if n_points <= 0 or n_points & (n_points - 1):
            self.fail(f"radix-2 error count requires a power-of-two FFT length, got {n_points}")
        pnts = int(self.EEG['pnts'])
        cutoff = (pnts // n_points) * n_points
        n_starts = np.ceil(np.arange(0, cutoff - n_points + 1, n_points / 2)).size
        n_seg = int(n_starts) * int(self.EEG['trials'])
        log2_n = n_points.bit_length() - 1
        median_add = 0 if n_seg % 2 else 1
        k = 1 + 8 * log2_n + 3 + (2 * n_points - 1) + 3 + median_add
        unit = float(np.finfo(np.float64).eps) / 2
        gamma = k * unit / (1.0 - k * unit)
        rtol = 2 * gamma
        atol = (20 / np.log(10)) * rtol

        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)
        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        EEG.icaact = double(EEG.icaact);
        psdmed = eeg_rpsd(EEG);
        icaact = EEG.icaact;
        save('{temp_file}.mat', 'psdmed', 'icaact');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        mat_data = scipy.io.loadmat(temp_file + '.mat', mat_dtype=True)
        activations = np.asarray(mat_data['icaact'])
        ml_result = mat_data['psdmed']
        compared = self.EEG.copy()
        compared['icaact'] = activations
        py_result = eeg_rpsd(compared)

        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        self.assertEqual(activations.dtype, np.dtype(np.float64))
        self.assertEqual(py_result.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(
            py_result,
            ml_result,
            rtol=rtol,
            atol=atol,
            err_msg="eeg_rpsd double-activation results differ beyond the float64 rounding bound",
        )


if __name__ == '__main__':
    unittest.main()
