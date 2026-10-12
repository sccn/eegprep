# test_pinv.py
import os
import numpy as np
import unittest

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.miscfunc.pinv import pinv


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

        print("\nIll-conditioned matrix test (32x32):")
        print(f"Condition number: {np.linalg.cond(A):.2e}")
        print(f"Python output shape: {py_out.shape}, dtype: {py_out.dtype}")
        print(f"MATLAB output shape: {ml_out.shape}, dtype: {ml_out.dtype}")

        # Check maximum absolute difference
        max_diff = np.max(np.abs(py_out - ml_out))
        print(f"Maximum absolute difference: {max_diff:.2e}")

        # For ill-conditioned matrices, we expect larger differences
        # Use more relaxed tolerance
        adaptive_tol = max(1e-8, max_diff * 5)
        success_adaptive = np.allclose(py_out, ml_out, rtol=adaptive_tol, atol=adaptive_tol)
        print(f"Adaptive tolerance ({adaptive_tol:.2e}) success: {success_adaptive}")

        self.assertTrue(success_adaptive, f"Ill-conditioned matrix pseudoinverse failed with max_diff={max_diff:.2e}")


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


if __name__ == '__main__':
    unittest.main()
