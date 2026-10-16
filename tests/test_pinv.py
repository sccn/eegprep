# test_pinv.py
import os
import warnings

import numpy as np
import unittest

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.miscfunc.pinv import pinv
from eegprep.plugins.clean_rawdata.private.riemann import asr_subspace_reconstruction


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestPinvParity(unittest.TestCase):
    """Test parity between Python pinv and MATLAB pinv functions."""

    def setUp(self):
        self.eeglab = get_eeglab('MAT')

    def test_parity_singular_matrix(self):
        """Test pseudoinverse of a singular matrix."""
        A = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=float)  # rank-deficient

        py_out = pinv(A)
        ml_out = self.eeglab.pinv(A)

        self.assertTrue(np.allclose(py_out, ml_out, atol=1e-12))

    def test_parity_with_custom_tolerance(self):
        """Test pseudoinverse with custom tolerance parameter."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        tol = 1e-10

        py_out = pinv(A, tol=tol)
        # MATLAB's pinv function uses 'tol' parameter: pinv(A, tol)
        ml_out = self.eeglab.pinv(A, tol)

        self.assertTrue(np.allclose(py_out, ml_out, atol=1e-12))

    def test_parity_ill_conditioned_matrix(self):
        """Compare Penrose products and the ASR projector, not pinv entries.

        Singular vectors tied to singular values a few times the cutoff are
        not identified across LAPACK builds, so element-wise ``pinv`` values
        are not a parity check. ``A @ pinv(A) @ A``, ``pinv(A) @ A @ pinv(A)``,
        and ``M @ pinv(V' M) @ V'`` are.
        """
        matrix = _ill_conditioned_matrix()
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        cutoff = max(matrix.shape) * np.spacing(singular_values[0])
        self.assertGreater(singular_values[-1], cutoff)

        python_inverse = pinv(matrix)
        matlab_inverse = np.asarray(self.eeglab.pinv(matrix), dtype=np.float64)
        bound = _projector_bound(singular_values)
        python_errors = _penrose_errors(matrix, python_inverse)
        matlab_errors = _penrose_errors(matrix, matlab_inverse)
        reconstruction_gap = np.linalg.norm(
            (matrix @ python_inverse @ matrix) - (matrix @ matlab_inverse @ matrix),
            2,
        )
        reconstruction_bound = 2.0 * bound * float(singular_values[0])
        asr_gap = _asr_reconstruction_gap(matrix, self.eeglab.pinv, np.ones(matrix.shape[0], dtype=bool))
        asr_bound = 2.0 * bound
        warnings.warn(
            "ill-conditioned pinv "
            f"projector py={python_errors[0]:.6e} ml={matlab_errors[0]:.6e} bound={bound:.6e} "
            f"recon py={python_errors[1]:.6e} ml={matlab_errors[1]:.6e} "
            f"bound={bound * singular_values[0]:.6e} "
            f"cross={reconstruction_gap:.6e} cross_bound={reconstruction_bound:.6e} "
            f"penrose py={python_errors[2]:.6e} ml={matlab_errors[2]:.6e} "
            f"bound={bound / singular_values[-1]:.6e} "
            f"asr_cross={asr_gap:.6e} bound={asr_bound:.6e}",
            UserWarning,
            stacklevel=1,
        )
        self._assert_penrose('python', python_errors, bound, singular_values)
        self._assert_penrose('matlab', matlab_errors, bound, singular_values)
        self.assertLessEqual(
            reconstruction_gap,
            reconstruction_bound,
            f"A@pinv@A cross={reconstruction_gap:.6e} bound={reconstruction_bound:.6e}",
        )
        self.assertLessEqual(
            asr_gap,
            asr_bound,
            f"ASR projector cross={asr_gap:.6e} bound={asr_bound:.6e}",
        )

    def _assert_penrose(self, label, errors, bound, singular_values):
        limits = (
            bound,
            bound * float(singular_values[0]),
            bound / float(singular_values[-1]),
        )
        names = ('projector', 'A@pinv@A', 'pinv@A@pinv')
        for name, error, limit in zip(names, errors, limits, strict=True):
            self.assertLessEqual(
                error,
                limit,
                f"{label} {name}={error:.6e} bound={limit:.6e}",
            )

    def test_parity_nonfinite_input(self):
        """NaN or Inf input matches MATLAB's all-NaN transposed result."""
        matrices = (
            np.array([[1.0, np.nan], [0.0, 2.0]]),
            np.array([[np.inf, 0.0, 1.0], [2.0, 3.0, 4.0]]),
        )
        for matrix in matrices:
            with self.subTest(matrix=matrix):
                python_inverse = pinv(matrix)
                matlab_inverse = np.asarray(self.eeglab.pinv(matrix), dtype=np.float64)
                self.assertEqual(python_inverse.shape, (matrix.shape[1], matrix.shape[0]))
                self.assertEqual(matlab_inverse.shape, python_inverse.shape)
                self.assertTrue(np.isnan(python_inverse).all())
                self.assertTrue(np.isnan(matlab_inverse).all())

    def test_parity_masked_rank_deficient_reconstruction(self):
        """Partial keep mask: the masked factor is the rank-deficient ASR path."""
        mixing, basis, keep = _masked_asr_case()
        factor = _masked_factor(mixing, basis, keep)
        singular_values = np.linalg.svd(factor, compute_uv=False)
        cutoff = max(factor.shape) * np.spacing(singular_values[0])
        numpy_cutoff = 1e-15 * singular_values[0]
        retained = singular_values[singular_values > cutoff]
        self.assertFalse(np.all(keep))
        self.assertGreater(np.count_nonzero((singular_values > numpy_cutoff) & (singular_values <= cutoff)), 0)
        self.assertGreater(retained.size, 0)

        gap = _asr_reconstruction_gap(mixing, self.eeglab.pinv, keep, basis)
        bound = 2.0 * _projector_bound(retained, order=mixing.shape[0])
        warnings.warn(
            f"masked asr_cross={gap:.6e} bound={bound:.6e}",
            UserWarning,
            stacklevel=1,
        )
        self.assertLessEqual(gap, bound, f"masked ASR cross={gap:.6e} bound={bound:.6e}")


class TestPinvFunctional(unittest.TestCase):
    """Functional tests for pinv without MATLAB comparison."""

    def test_pinv_properties(self):
        """Test mathematical properties of pseudoinverse."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        A_pinv = pinv(A)

        # Test A * A_pinv * A = A (within numerical precision)
        self.assertTrue(np.allclose(A @ A_pinv @ A, A, atol=1e-12))

        # Test A_pinv * A * A_pinv = A_pinv (within numerical precision)
        self.assertTrue(np.allclose(A_pinv @ A @ A_pinv, A_pinv, atol=1e-12))

    def test_explicit_tol_is_an_absolute_cutoff(self):
        """tol is MATLAB's absolute singular-value cutoff, not a relative rtol."""
        matrix = np.diag([100.0, 0.05])

        inverse = pinv(matrix, tol=1e-3)
        # 0.05 is below a relative cutoff of 1e-3 * 100 and above the absolute cutoff 1e-3.
        np.testing.assert_allclose(inverse, np.diag([0.01, 20.0]), atol=1e-12)

    def test_nonfinite_input_is_all_nan(self):
        """NaN or Inf input returns all NaN with the transposed shape."""
        nan_matrix = np.array([[1.0, np.nan], [0.0, 2.0]])
        inf_matrix = np.array([[np.inf, 0.0, 1.0], [2.0, 3.0, 4.0]])
        for matrix in (nan_matrix, inf_matrix):
            with self.subTest(matrix=matrix):
                inverse = pinv(matrix)
                self.assertEqual(inverse.shape, (matrix.shape[1], matrix.shape[0]))
                self.assertTrue(np.isnan(inverse).all())

    def test_ill_conditioned_products_meet_the_projector_bound(self):
        """The derived projector bound covers this matrix and rejects a rank drop."""
        matrix = _ill_conditioned_matrix()
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        cutoff = max(matrix.shape) * np.spacing(singular_values[0])
        self.assertGreater(singular_values[-1], cutoff)
        bound = _projector_bound(singular_values)
        errors = _penrose_errors(matrix, pinv(matrix))
        self.assertLessEqual(errors[0], bound)
        self.assertLessEqual(errors[1], bound * singular_values[0])
        self.assertLessEqual(errors[2], bound / singular_values[-1])

        dropped = pinv(matrix, tol=float(np.sqrt(singular_values[-1] * singular_values[-2])))
        dropped_projector = np.linalg.norm(matrix @ dropped - np.eye(matrix.shape[0]), 2)
        self.assertGreater(dropped_projector, bound)

    def test_partial_keep_drops_the_masked_null(self):
        """A partial keep mask still drops a singular value NumPy's cutoff would invert."""
        mixing, basis, keep = _masked_asr_case()
        factor = _masked_factor(mixing, basis, keep)
        singular_values = np.linalg.svd(factor, compute_uv=False)
        cutoff = max(factor.shape) * np.spacing(singular_values[0])
        numpy_cutoff = 1e-15 * singular_values[0]
        self.assertFalse(np.all(keep))
        self.assertGreater(np.count_nonzero((singular_values > numpy_cutoff) & (singular_values <= cutoff)), 0)

        retained = singular_values[singular_values > cutoff]
        reconstructed = asr_subspace_reconstruction(mixing, basis, keep)
        numpy_reconstructed = np.real(mixing @ np.linalg.pinv(factor) @ np.asarray(basis).T)
        np.testing.assert_allclose(np.linalg.norm(pinv(factor), 2), 1.0 / retained[-1], rtol=1e-6, atol=0.0)
        self.assertGreater(np.linalg.norm(reconstructed - numpy_reconstructed, 2), 0.5)

    def test_default_cutoff_matches_matlab_spacing_at_the_norm(self):
        """Keep a singular value between MATLAB's cutoff and SciPy's relative default."""
        matrix = _spectrum_matrix(5e-13)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        cutoff = max(matrix.shape) * np.spacing(singular_values[0])
        scipy_cutoff = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
        self.assertGreater(singular_values[-1], cutoff)
        self.assertLess(singular_values[-1], scipy_cutoff)

        inverse = pinv(matrix)

        np.testing.assert_allclose(np.linalg.norm(inverse, 2), 1.0 / singular_values[-1], rtol=1e-6, atol=0.0)

    def test_default_cutoff_drops_a_numpy_null_singular_value(self):
        """Drop a numerical null that NumPy's 1e-15 cutoff would invert."""
        matrix = _spectrum_matrix(2e-13)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        cutoff = max(matrix.shape) * np.spacing(singular_values[0])
        numpy_cutoff = 1e-15 * singular_values[0]
        self.assertLess(singular_values[-1], cutoff)
        self.assertGreater(singular_values[-1], numpy_cutoff)
        retained = singular_values[singular_values > cutoff]

        inverse = pinv(matrix)

        np.testing.assert_allclose(np.linalg.norm(inverse, 2), 1.0 / retained[-1], rtol=1e-6, atol=0.0)
        self.assertLess(np.linalg.norm(inverse, 2), 1.0 / cutoff)


def _projector_bound(singular_values, order=None):
    """Return ``sqrt(n) * eps * cond(A)`` for ``||A @ pinv(A) - I||_2``.

    A backward-stable SVD is exact for a matrix within one unit roundoff of
    ``A``. That perturbation moves the projector by ``eps * cond(A)``. The
    products that form the projector are length ``n``, and independent
    roundoff adds in root-sum-square, which contributes ``sqrt(n)``.
    """
    count = singular_values.shape[0] if order is None else order
    condition = float(singular_values[0] / singular_values[-1])
    return float(np.sqrt(count) * np.finfo(np.float64).eps * condition)


def _penrose_errors(matrix, inverse):
    eye = np.eye(matrix.shape[0], dtype=matrix.dtype)
    projector = float(np.linalg.norm(matrix @ inverse - eye, 2))
    reconstructed = float(np.linalg.norm(matrix @ inverse @ matrix - matrix, 2))
    penrose = float(np.linalg.norm(inverse @ matrix @ inverse - inverse, 2))
    return projector, reconstructed, penrose


def _asr_reconstruction_gap(matrix, matlab_pinv, keep, basis=None):
    """``||R_python - R_matlab||_2`` for ``R = M @ pinv(keep .* (V' M)) @ V'``."""
    if basis is None:
        basis, _ = np.linalg.qr(np.random.default_rng(306).standard_normal((matrix.shape[0], matrix.shape[0])))
    python_reconstruction = asr_subspace_reconstruction(matrix, basis, keep)
    factor = _masked_factor(matrix, basis, keep)
    matlab_inverse = np.asarray(matlab_pinv(factor), dtype=np.float64)
    matlab_reconstruction = np.real(matrix @ matlab_inverse @ np.asarray(basis).T)
    return float(np.linalg.norm(python_reconstruction - matlab_reconstruction, 2))


def _masked_factor(mixing, basis, keep):
    projected = np.asarray(basis).T @ mixing
    return projected * np.asarray(keep, dtype=float)[:, None]


def _masked_asr_case():
    """Mixing whose kept rows include one singular value below MATLAB's cutoff."""
    channels = 62
    kept = channels - 4
    rng = np.random.default_rng(306)
    singular_values = np.linspace(45.11, 1.42, kept)
    singular_values[-1] = 2e-13
    left, _ = np.linalg.qr(rng.standard_normal((kept, kept)))
    right, _ = np.linalg.qr(rng.standard_normal((channels, kept)))
    basis, _ = np.linalg.qr(rng.standard_normal((channels, channels)))
    projected = rng.standard_normal((channels, channels))
    projected[:kept] = (left * singular_values) @ right.T
    keep = np.zeros(channels, dtype=bool)
    keep[:kept] = True
    return basis @ projected, basis, keep


def _ill_conditioned_matrix():
    rng = np.random.default_rng(99999)
    matrix = rng.standard_normal((32, 32))
    left, singular_values, right = np.linalg.svd(matrix)
    singular_values[20:] *= 1e-12
    return left @ np.diag(singular_values) @ right


def _spectrum_matrix(smallest, channels=62):
    singular_values = np.linspace(45.11, 1.42, channels)
    singular_values[-1] = smallest
    left, _ = np.linalg.qr(np.random.default_rng(306).standard_normal((channels, channels)))
    right, _ = np.linalg.qr(np.random.default_rng(457).standard_normal((channels, channels)))
    return left @ np.diag(singular_values) @ right.T


if __name__ == '__main__':
    unittest.main()
