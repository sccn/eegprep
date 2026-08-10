"""Numerical parity between EEGPrep and MATLAB/EEGLAB ``envtopo``.

Runs EEGLAB's ``envtopo`` through the MATLAB engine and compares the returned
component-contribution outputs (ranking, selection, per-component peak frame and
latency, and the sort-metric values) against EEGPrep's ``envtopo`` on the
identical epoched ICA dataset. Requires the MATLAB engine plus an EEGLAB checkout
(via ``eeglabcompat.get_eeglab``) and is skipped when either is unavailable, e.g.
in CI. No MATLAB output is committed; the reference is regenerated live, mirroring
``test_spectopo_parity.py``.

Comparisons target the well-defined, quirk-free quantities:
  * ``compsplotted`` -- the top-N plotted IC numbers (distinct large metrics, so
    ordering is stable) -- compared exactly.
  * the candidate set of ``compvarorder`` and its top-N prefix.
  * the multiset of ``sortvar`` values (order-free, so it holds across sort modes
    and EEGLAB's candidate-vs-sorted ordering).
  * per plotted IC, the peak frame and latency, aligned by IC number.
We deliberately do not compare the full ``compvarorder`` tail exactly: on real
data many low-power components have near-equal metrics, so their relative order is
ambiguous between MATLAB's and NumPy's sorts. Boundaries under test: EEGPrep
returns 0-based ``compframes`` (MATLAB is 1-based) and ``comptimes`` in ms
(MATLAB in seconds).
"""

# Force single-threaded BLAS for deterministic numerics (mirrors
# test_spectopo_parity); must be set before numpy imports.
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
import scipy.io

from eegprep import pop_loadset, pop_saveset
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.sigprocfunc.envtopo import envtopo

local_url = os.path.join(os.path.dirname(__file__), "../sample_data/")

# Float tolerance for sort-metric values and latencies. Observed max |Δ| is well
# below this on the sample data; it guards against real miscalibration without
# float flakiness.
RTOL = 1e-6
ATOL = 1e-9


def _icol(value):
    return np.asarray(value, dtype=float).ravel().astype(int)


def _fcol(value):
    return np.asarray(value, dtype=float).ravel()


def _matlab_value(value):
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + " ".join(_matlab_scalar(item) for item in np.asarray(value).ravel()) + "]"
    return _matlab_scalar(value)


def _matlab_scalar(value):
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)


def _matlab_options(options):
    return "".join(f", '{key}', {_matlab_value(value)}" for key, value in options.items())


class TestEnvtopoParity(unittest.TestCase):
    """Parity between Python and MATLAB envtopo component-contribution outputs."""

    def setUp(self):
        try:
            self.eeglab = get_eeglab("MAT", auto_file_roundtrip=False)
        except Exception as e:
            self.skipTest(f"MATLAB/EEGLAB not available: {e}")
        # Epoched ICA dataset: envtopo averages epochs and ranks IC back-projections.
        self.EEG = pop_loadset(os.path.join(local_url, "eeglab_data_epochs_ica.set"))
        data = np.asarray(self.EEG["data"], dtype=float)
        self.sig = data.mean(axis=2) if data.ndim == 3 else data
        # finite_matmul matches plain matmul but suppresses the spurious FPE warnings
        # the arm64 BLAS raises on this (finite) product, as plot_utils does for ICA.
        self.weights = finite_matmul(
            np.asarray(self.EEG["icaweights"], dtype=float), np.asarray(self.EEG["icasphere"], dtype=float)
        )
        self.icawinv = np.asarray(self.EEG["icawinv"], dtype=float)
        self.chanlocs = self.EEG["chanlocs"]
        self.timerange = [float(self.EEG["xmin"]) * 1000.0, float(self.EEG["xmax"]) * 1000.0]

    def _matlab_outputs(self, options):
        """Run EEGLAB envtopo on the saved dataset and return its six outputs."""
        temp_file = tempfile.mktemp(suffix=".set")
        pop_saveset(self.EEG, temp_file)
        matlab_code = f"""
        set(0, 'DefaultFigureVisible', 'off');
        EEG = pop_loadset('{temp_file}');
        sig = mean(double(EEG.data), 3);  % double: EEG.data is single, EEGPrep computes in float64
        [compvarorder, compvars, compframes, comptimes, compsplotted, sortvar] = envtopo( ...
            sig, EEG.icaweights*EEG.icasphere, 'chanlocs', EEG.chanlocs, 'icawinv', EEG.icawinv, ...
            'timerange', [{self.timerange[0]} {self.timerange[1]}]{_matlab_options(options)});
        close all; set(0, 'DefaultFigureVisible', 'on');
        save('{temp_file}.mat', 'compvarorder', 'compvars', 'compframes', 'comptimes', 'compsplotted', 'sortvar');
        """
        self.eeglab.eval(matlab_code, nargout=0)
        mat = scipy.io.loadmat(temp_file + ".mat")
        os.remove(temp_file)
        os.remove(temp_file + ".mat")
        fdt = temp_file.replace(".set", ".fdt")
        if os.path.exists(fdt):
            os.remove(fdt)
        return mat

    def _assert_case(self, options):
        mat = self._matlab_outputs(options)
        ml_order = _icol(mat["compvarorder"])
        ml_plotted = _icol(mat["compsplotted"])
        ml_sortvar = _fcol(mat["sortvar"])
        ml_frames = _icol(mat["compframes"])  # 1-based absolute frames
        times_ms = np.linspace(self.timerange[0], self.timerange[1], self.sig.shape[1])

        res = envtopo(
            self.sig,
            self.weights,
            chanlocs=self.chanlocs,
            icawinv=self.icawinv,
            timerange=self.timerange,
            **options,
        )
        py_order = np.asarray(res.compvarorder, dtype=int)
        py_plotted = np.asarray(res.compsplotted, dtype=int)

        # Top-N plotted ICs: exact and order-sensitive.
        np.testing.assert_array_equal(py_plotted, ml_plotted, err_msg=f"compsplotted differ for {options}")
        # Candidate set matches, and the ranking's top-N prefix equals compsplotted.
        self.assertEqual(set(py_order.tolist()), set(ml_order.tolist()), msg=f"candidate set differs for {options}")
        np.testing.assert_array_equal(py_order[: ml_plotted.size], ml_plotted)
        # Sort-metric values: order-free multiset comparison.
        np.testing.assert_allclose(
            np.sort(np.asarray(res.sortvar, dtype=float)),
            np.sort(ml_sortvar),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"sortvar multiset differs for {options}",
        )
        # Peak frame (0-based) and latency (ms) per plotted IC, aligned by IC number.
        py_frame = dict(zip(py_order.tolist(), np.asarray(res.compframes, dtype=int).tolist()))
        py_time = dict(zip(py_order.tolist(), np.asarray(res.comptimes, dtype=float).tolist()))
        ml_frame = dict(zip(ml_order.tolist(), (ml_frames - 1).tolist()))
        for ic in ml_plotted.tolist():
            self.assertEqual(py_frame[ic], ml_frame[ic], msg=f"peak frame differs for IC{ic}, {options}")
            # EEGLAB's comptimes output double-applies the sort permutation (envtopo.m:731) and is
            # misaligned; derive the expected latency from MATLAB's correct compframes instead.
            np.testing.assert_allclose(
                py_time[ic],
                times_ms[ml_frame[ic]],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"peak latency differs for IC{ic}, {options}",
            )
        plt.close(res.figure)

    def test_parity_sortvar_modes(self):
        """All four ranking modes match EEGLAB (mp default, plus pv/pp/rp)."""
        for mode in ("mp", "pv", "pp", "rp"):
            with self.subTest(sortvar=mode):
                self._assert_case({"sortvar": mode})

    def test_parity_compnums_subset(self):
        """Ranking restricted to an explicit candidate list."""
        self._assert_case({"compnums": [1, 3, 5, 7, 9, 11]})

    def test_parity_compsplot_count(self):
        """Plotting fewer than the default number of components."""
        self._assert_case({"compsplot": 3})

    def test_parity_limcontrib_window(self):
        """Ranking window narrower than the epoch."""
        span = self.timerange[1] - self.timerange[0]
        limcontrib = [self.timerange[0] + 0.3 * span, self.timerange[1] - 0.3 * span]
        self._assert_case({"limcontrib": limcontrib})

    def test_parity_subcomps_subtracted(self):
        """Subtracted components are removed before ranking, matching EEGLAB."""
        self._assert_case({"subcomps": [2, 4]})


if __name__ == "__main__":
    unittest.main()
