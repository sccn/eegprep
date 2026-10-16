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
        asr_gap = _asr_reconstruction_gap(matrix, self.eeglab.pinv)
        asr_bound = 2.0 * bound
        warnings.warn(
            "ill-conditioned pinv "
            f"projector py={python_errors[0]:.6e} ml={matlab_errors[0]:.6e} bound={bound:.6e} "
            f"recon py={python_errors[1]:.6e} ml={matlab_errors[1]:.6e} "
            f"bound={bound * singular_values[0]:.6e} cross={reconstruction_gap:.6e} "
            f"penrose py={python_errors[2]:.6e} ml={matlab_errors[2]:.6e} "
            f"bound={bound / singular_values[-1]:.6e} "
            f"asr_cross={asr_gap:.6e} bound={asr_bound:.6e}",
            UserWarning,
            stacklevel=2,
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


def _projector_bound(singular_values):
    """Return ``sqrt(n) * eps * cond(A)`` for ``||A @ pinv(A) - I||_2``.

    A backward-stable SVD is exact for a matrix within one unit roundoff of
    ``A``. That perturbation moves the projector by ``eps * cond(A)``. The
    products that form the projector are length ``n``, and independent
    roundoff adds in root-sum-square, which contributes ``sqrt(n)``.
    """
    order = singular_values.shape[0]
    condition = float(singular_values[0] / singular_values[-1])
    return float(np.sqrt(order) * np.finfo(np.float64).eps * condition)


def _penrose_errors(matrix, inverse):
    eye = np.eye(matrix.shape[0], dtype=matrix.dtype)
    projector = float(np.linalg.norm(matrix @ inverse - eye, 2))
    reconstructed = float(np.linalg.norm(matrix @ inverse @ matrix - matrix, 2))
    penrose = float(np.linalg.norm(inverse @ matrix @ inverse - inverse, 2))
    return projector, reconstructed, penrose


def _asr_reconstruction_gap(matrix, matlab_pinv):
    """``||R_python - R_matlab||_2`` for ``R = M @ pinv(V' M) @ V'``."""
    order = matrix.shape[0]
    basis, _ = np.linalg.qr(np.random.default_rng(306).standard_normal((order, order)))
    python_reconstruction = asr_subspace_reconstruction(matrix, basis, np.ones(order, dtype=bool))
    factor = basis.T @ matrix
    matlab_inverse = np.asarray(matlab_pinv(factor), dtype=np.float64)
    matlab_reconstruction = np.real(matrix @ matlab_inverse @ basis.T)
    return float(np.linalg.norm(python_reconstruction - matlab_reconstruction, 2))


def _ill_conditioned_matrix():
    np.random.seed(99999)
    matrix = np.random.randn(32, 32).astype(np.float64)
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
