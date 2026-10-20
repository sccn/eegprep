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
from numpy.fft import fft
from eegprep import pop_loadset, pop_saveset, eeg_rpsd
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.plugins.firfilt.windows import windows

local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')


def rpsd_double_bound_db(activations, srate, pnts, psd_reference):
    """Return the per-bin decibel separation allowed for two float64 spectra."""
    u = float(np.finfo(np.float64).eps) / 2

    def gamma(k):
        return k * u / (1 - k * u)

    n = min(int(pnts), int(srate))
    log2n = n.bit_length() - 1
    nyquist = int(srate) // 2
    eta = u + gamma(4) * (np.sqrt(2) + u)
    fft_rel = log2n * eta / (1 - log2n * eta)
    window = windows("hamming", n)
    cutoff = (int(pnts) // n) * n
    index = np.add.outer(np.ceil(np.arange(0, cutoff - n + 1, n / 2)).astype(int), np.arange(n)).astype(int).transpose()
    scale = 2 / (srate * np.sum(window**2))
    bound = np.empty(np.shape(psd_reference), dtype=np.float64)
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim == 3:
        values = np.reshape(values, (values.shape[0], -1), order="F")
    for component in range(values.shape[0]):
        segments = values[component, index].T * window
        magnitude = np.abs(fft(segments, axis=1)[:, 1 : nyquist + 1])
        beta = ((u + fft_rel) * np.sqrt(n) * np.linalg.norm(segments, axis=1))[:, None]
        power = magnitude**2 * scale
        delta = scale * (2 * magnitude * beta + beta**2) + gamma(2 * n + 5) * scale * (magnitude + beta) ** 2
        power[:, -1] /= 2
        delta[:, -1] /= 2
        low = np.median(np.maximum(power - delta, 0), axis=0)
        high = np.median(power + delta, axis=0)
        reference = np.abs(np.asarray(psd_reference[component], dtype=np.float64))
        bound[component] = (20 / np.log(10)) * np.log(high / low) + 6 * u * reference
    return bound


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

        ``pop_loadset`` recomputes the activations and MATLAB ``double``
        casts them. Both sides window that array. The default keeps every
        segment, so the median does not depend on permutation order.

        Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
        Theorem 24.2, bounds a radix-2 FFT by
        ``||fl(Fx) - Fx||_2 <= (log2(n) * eta / (1 - log2(n) * eta)) * ||Fx||_2``,
        with ``eta = u + gamma_4 * (sqrt(2) + u)``, ``u = eps(float64) / 2``,
        ``gamma_k = k u / (1 - k u)``, and ``||Fx||_2 = sqrt(n) * ||x||_2``.
        A bin's error is bounded by that factor times the segment norm, not
        by ``|X_k|``. FFTW and NumPy's pocketfft are not radix-2
        Cooley-Tukey; their bounds have this same form. The window multiply
        adds a relative ``u`` on the segment, carried through ``||F||`` as
        another ``u * sqrt(n) * ||x||``. ``gamma_{2n+5}`` is the relative
        rounding of ``sum(window.^2)`` (``2n - 1`` flops), the multiply by
        the sampling rate, the division, the multiply by that scale, and
        ``|z|^2`` (3 flops). Dividing the Nyquist bin by 2 is exact, so its
        absolute error halves. The median is monotone, so each
        implementation's median lies between the medians of ``power - delta``
        and ``power + delta``. In decibels that gap is
        ``(20/ln(10)) * ln(high/low)``. Each ``20*log10`` adds three unit
        roundoffs (log, division by ``ln(10)``, multiply by 20), and the two
        evaluations add ``6 u |result|``. The assertion is
        ``max(|python - matlab| - bound) <= 0``.
        """
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        n_points = min(int(self.EEG["pnts"]), int(self.EEG["srate"]))
        if n_points <= 0 or n_points & (n_points - 1):
            self.fail(f"Theorem 24.2 needs a power-of-two FFT length, got {n_points}")

        temp_file = tempfile.mktemp(suffix=".set")
        pop_saveset(self.EEG, temp_file)
        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        EEG.icaact = double(EEG.icaact);
        psdmed = eeg_rpsd(EEG);
        icaact = EEG.icaact;
        save('{temp_file}.mat', 'psdmed', 'icaact');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        mat_data = scipy.io.loadmat(temp_file + ".mat", mat_dtype=True)
        activations = np.asarray(mat_data["icaact"])
        ml_result = np.asarray(mat_data["psdmed"])
        compared = self.EEG.copy()
        compared["icaact"] = activations
        py_result = eeg_rpsd(compared)

        os.remove(temp_file)
        os.remove(temp_file + ".mat")
        if os.path.exists(temp_file.replace(".set", ".fdt")):
            os.remove(temp_file.replace(".set", ".fdt"))

        self.assertEqual(activations.dtype, np.dtype(np.float64))
        self.assertEqual(py_result.dtype, np.dtype(np.float64))
        bound = rpsd_double_bound_db(activations, self.EEG["srate"], self.EEG["pnts"], ml_result)
        excess = np.abs(py_result - ml_result) - bound
        worst = float(np.max(excess))
        where = np.unravel_index(int(np.argmax(excess)), excess.shape)
        self.assertLessEqual(
            worst,
            0.0,
            msg=f"worst excess {worst} at component {where[0]}, bin {where[1]}",
        )


if __name__ == '__main__':
    unittest.main()
