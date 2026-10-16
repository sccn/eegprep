# test_pinv.py
import os
import numpy as np
import unittest

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.miscfunc.pinv import pinv

# Seeded 32x32 matrix in test_parity_ill_conditioned_matrix: its smallest
# singular value is about 2.57 times MATLAB's cutoff max(size)*eps(norm(A)).
# An SVD backward error of n*eps*||A|| changes 1/sigma by the relative amount
# n*eps*||A||/sigma_min ≈ 0.523. The matching absolute tolerance is that
# relative amount divided by sigma_min.
ILL_CONDITIONED_RTOL = 0.53


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
        """Test pseudoinverse of an ill-conditioned matrix."""
        np.random.seed(99999)
        A = np.random.randn(32, 32).astype(np.float64)

        # Make it ill-conditioned by scaling some singular values very small
        U, s, Vt = np.linalg.svd(A)
        s[20:] *= 1e-12  # Make some singular values very small
        A = U @ np.diag(s) @ Vt

        py_out = pinv(A)
        ml_out = self.eeglab.pinv(A)

        singular_values = np.linalg.svd(A, compute_uv=False)
        cutoff = max(A.shape) * np.spacing(singular_values[0])
        sensitivity = max(A.shape) * np.finfo(float).eps * singular_values[0] / singular_values[-1]
        self.assertGreater(singular_values[-1], cutoff)
        self.assertLess(sensitivity, ILL_CONDITIONED_RTOL)

        max_diff = np.max(np.abs(py_out - ml_out))
        absolute_bound = ILL_CONDITIONED_RTOL / singular_values[-1]
        self.assertTrue(
            np.allclose(py_out, ml_out, rtol=ILL_CONDITIONED_RTOL, atol=absolute_bound),
            f"Ill-conditioned matrix pseudoinverse failed with max_diff={max_diff:.3e} "
            f"atol={absolute_bound:.3e} rtol={ILL_CONDITIONED_RTOL}",
        )


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

        for method in ('scipy', 'svd'):
            with self.subTest(method=method):
                inverse = pinv(matrix, tol=1e-3, method=method)
                # 0.05 is below a relative cutoff of 1e-3 * 100 and above the absolute cutoff 1e-3.
                np.testing.assert_allclose(inverse, np.diag([0.01, 20.0]), atol=1e-12)

    def test_ill_conditioned_bound_covers_cutoff_sensitivity(self):
        """The fixed parity rtol stays above this matrix's cutoff sensitivity."""
        np.random.seed(99999)
        matrix = np.random.randn(32, 32).astype(np.float64)
        left, singular_values, right = np.linalg.svd(matrix)
        singular_values[20:] *= 1e-12
        matrix = left @ np.diag(singular_values) @ right

        singular_values = np.linalg.svd(matrix, compute_uv=False)
        cutoff = max(matrix.shape) * np.spacing(singular_values[0])
        sensitivity = max(matrix.shape) * np.finfo(float).eps * singular_values[0] / singular_values[-1]

        self.assertGreater(singular_values[-1], cutoff)
        self.assertLess(sensitivity, ILL_CONDITIONED_RTOL)

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


def _spectrum_matrix(smallest, channels=62):
    singular_values = np.linspace(45.11, 1.42, channels)
    singular_values[-1] = smallest
    left, _ = np.linalg.qr(np.random.default_rng(306).standard_normal((channels, channels)))
    right, _ = np.linalg.qr(np.random.default_rng(457).standard_normal((channels, channels)))
    return left @ np.diag(singular_values) @ right.T


if __name__ == '__main__':
    unittest.main()
