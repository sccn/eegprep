import unittest
import numpy as np

from eegprep.plugins.clean_rawdata.private.covariance import (
    cov_logm,
    cov_expm,
    cov_powm,
    cov_sqrtm,
    cov_rsqrtm,
    cov_sqrtm2,
    cov_mean,
    cov_shrinkage,
)


class TestCovarianceMatrixOperations(unittest.TestCase):
    """Test matrix operations on covariance matrices."""

    def setUp(self):
        """Create test covariance matrices."""
        # Simple 2x2 positive definite matrix
        self.cov_2x2 = np.array([[2.0, 1.0], [1.0, 2.0]])

        # 3x3 positive definite matrix
        self.cov_3x3 = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 1.0], [0.5, 1.0, 2.0]])

        # Stack of covariance matrices
        self.cov_stack = np.array([[[2.0, 1.0], [1.0, 2.0]], [[3.0, 0.5], [0.5, 1.5]]])

    def test_cov_logm_single_matrix(self):
        """Test matrix logarithm of single covariance matrix."""
        result = cov_logm(self.cov_2x2)

        # Verify result is symmetric
        np.testing.assert_array_almost_equal(result, result.T, decimal=10)

        # Verify expm(logm(C)) = C
        reconstructed = cov_expm(result)
        np.testing.assert_array_almost_equal(reconstructed, self.cov_2x2, decimal=10)

    def test_cov_powm_single_matrix(self):
        """Test matrix power operation."""
        # Test square root (power = 0.5)
        sqrt_result = cov_powm(self.cov_2x2, 0.5)

        # sqrt(C) @ sqrt(C) should equal C
        reconstructed = sqrt_result @ sqrt_result
        np.testing.assert_array_almost_equal(reconstructed, self.cov_2x2, decimal=10)

        # Test square (power = 2)
        square_result = cov_powm(self.cov_2x2, 2.0)
        expected = self.cov_2x2 @ self.cov_2x2
        np.testing.assert_array_almost_equal(square_result, expected, decimal=10)

    def test_cov_rsqrtm_single_matrix(self):
        """Test matrix reciprocal square root."""
        result = cov_rsqrtm(self.cov_2x2)

        # rsqrt(C) @ C @ rsqrt(C) should equal identity
        whitened = result @ self.cov_2x2 @ result
        np.testing.assert_array_almost_equal(whitened, np.eye(2), decimal=10)

        # Result should be symmetric and positive definite
        np.testing.assert_array_almost_equal(result, result.T, decimal=10)
        eigenvals = np.linalg.eigvals(result)
        self.assertTrue(np.all(eigenvals > 0))

    def test_stack_operations(self):
        """Test operations on stacks of covariance matrices."""
        # Test all operations work with stacks
        log_stack = cov_logm(self.cov_stack)
        exp_stack = cov_expm(log_stack)
        sqrt_stack = cov_sqrtm(self.cov_stack)
        rsqrt_stack = cov_rsqrtm(self.cov_stack)
        pow_stack = cov_powm(self.cov_stack, 0.5)
        sqrt2_stack, rsqrt2_stack = cov_sqrtm2(self.cov_stack)

        # Check shapes
        self.assertEqual(log_stack.shape, self.cov_stack.shape)
        self.assertEqual(exp_stack.shape, self.cov_stack.shape)
        self.assertEqual(sqrt_stack.shape, self.cov_stack.shape)
        self.assertEqual(rsqrt_stack.shape, self.cov_stack.shape)

        # Check round-trip: exp(log(C)) = C
        np.testing.assert_array_almost_equal(exp_stack, self.cov_stack, decimal=10)

        # Check sqrt consistency
        np.testing.assert_array_almost_equal(sqrt_stack, pow_stack, decimal=10)
        np.testing.assert_array_almost_equal(sqrt_stack, sqrt2_stack, decimal=10)
        np.testing.assert_array_almost_equal(rsqrt_stack, rsqrt2_stack, decimal=10)


class TestCovMean(unittest.TestCase):
    """Test the covariance mean function."""

    def setUp(self):
        """Create test data."""
        # Create a stack of similar covariance matrices
        self.cov_stack = np.array([[[2.0, 0.5], [0.5, 1.5]], [[2.2, 0.3], [0.3, 1.8]], [[1.8, 0.7], [0.7, 1.2]]])

        # Single matrix (should return itself)
        self.single_cov = np.array([[[3.0, 1.0], [1.0, 2.0]]])

    def test_weighted_mean(self):
        """Test weighted mean of covariance matrices."""
        weights = np.array([0.5, 0.3, 0.2])
        result = cov_mean(self.cov_stack, weights=weights)

        # Result should be symmetric and positive definite
        np.testing.assert_array_almost_equal(result, result.T, decimal=10)
        eigenvals = np.linalg.eigvals(result)
        self.assertTrue(np.all(eigenvals > 0))

        # Should be closer to the first matrix (highest weight)
        dist_to_first = np.linalg.norm(result - self.cov_stack[0])
        dist_to_last = np.linalg.norm(result - self.cov_stack[2])
        self.assertLess(dist_to_first, dist_to_last)


class TestCovShrinkage(unittest.TestCase):
    """Test the covariance shrinkage function."""

    def setUp(self):
        """Create test covariance matrices."""
        self.cov_2x2 = np.array([[4.0, 2.0], [2.0, 3.0]])
        self.cov_3x3 = np.array([[5.0, 1.0, 0.5], [1.0, 4.0, 1.5], [0.5, 1.5, 3.0]])

        # Stack of matrices
        self.cov_stack = np.array([[[4.0, 2.0], [2.0, 3.0]], [[6.0, 1.0], [1.0, 2.0]]])

    def test_partial_shrinkage(self):
        """Test partial shrinkage."""
        shrinkage = 0.3
        result = cov_shrinkage(self.cov_2x2, shrinkage=shrinkage, target='eye')

        # Should be weighted combination
        expected = shrinkage * np.eye(2) + (1 - shrinkage) * self.cov_2x2
        np.testing.assert_array_almost_equal(result, expected)

        # Result should still be positive definite
        eigenvals = np.linalg.eigvals(result)
        self.assertTrue(np.all(eigenvals > 0))

    def test_scaled_eye_with_stack(self):
        """Test scaled-eye target with stack of matrices."""
        result = cov_shrinkage(self.cov_stack, shrinkage=1.0, target='scaled-eye')

        # Each matrix should be scaled identity
        for i in range(len(self.cov_stack)):
            trace = np.trace(self.cov_stack[i])
            scale = trace / 2
            expected = scale * np.eye(2)
            np.testing.assert_array_almost_equal(result[i], expected)


if __name__ == '__main__':
    unittest.main()
