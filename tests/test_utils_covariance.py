import unittest
import numpy as np
import warnings
from unittest.mock import patch

from eegprep.utils.covariance import (
    diag_nd, cov_logm, cov_expm, cov_powm, cov_sqrtm, cov_rsqrtm, 
    cov_sqrtm2, cov_mean, cov_shrinkage
)


class TestDiagNd(unittest.TestCase):
    """Test the diag_nd utility function."""
    
    def test_1d_input(self):
        """Test diag_nd with 1D input (like np.diag)."""
        x = np.array([1, 2, 3])
        result = diag_nd(x)
        expected = np.diag(x)
        np.testing.assert_array_equal(result, expected)
    
    def test_2d_input(self):
        """Test diag_nd with 2D input (multiple diagonal matrices)."""
        x = np.array([[1, 2], [3, 4]])  # Shape: (2, 2)
        result = diag_nd(x)
        
        # Should create 2 diagonal matrices
        expected = np.array([
            [[1, 0], [0, 2]],  # diag([1, 2])
            [[3, 0], [0, 4]]   # diag([3, 4])
        ])
        np.testing.assert_array_equal(result, expected)
    
    def test_3d_input(self):
        """Test diag_nd with 3D input."""
        x = np.array([[[1, 2, 3]], [[4, 5, 6]]])  # Shape: (2, 1, 3)
        result = diag_nd(x)
        
        # Should have shape (2, 1, 3, 3)
        self.assertEqual(result.shape, (2, 1, 3, 3))
        
        # Check first diagonal matrix
        expected_first = np.diag([1, 2, 3])
        np.testing.assert_array_equal(result[0, 0], expected_first)


class TestCovarianceMatrixOperations(unittest.TestCase):
    """Test matrix operations on covariance matrices."""
    
    def setUp(self):
        """Create test covariance matrices."""
        # Simple 2x2 positive definite matrix
        self.cov_2x2 = np.array([[2.0, 1.0], [1.0, 2.0]])
        
        # 3x3 positive definite matrix
        self.cov_3x3 = np.array([
            [4.0, 1.0, 0.5],
            [1.0, 3.0, 1.0],
            [0.5, 1.0, 2.0]
        ])
        
        # Stack of covariance matrices
        self.cov_stack = np.array([
            [[2.0, 1.0], [1.0, 2.0]],
            [[3.0, 0.5], [0.5, 1.5]]
        ])
    
    def test_cov_logm_single_matrix(self):
        """Test matrix logarithm of single covariance matrix."""
        result = cov_logm(self.cov_2x2)
        
        # Verify result is symmetric
        np.testing.assert_array_almost_equal(result, result.T, decimal=10)
        
        # Verify expm(logm(C)) = C
        reconstructed = cov_expm(result)
        np.testing.assert_array_almost_equal(reconstructed, self.cov_2x2, decimal=10)
    
    def test_cov_expm_single_matrix(self):
        """Test matrix exponential of single matrix."""
        # Start with log of a matrix, then exponentiate
        log_cov = cov_logm(self.cov_2x2)
        result = cov_expm(log_cov)
        
        # Should recover original matrix
        np.testing.assert_array_almost_equal(result, self.cov_2x2, decimal=10)
        
        # Result should be positive definite (all eigenvalues > 0)
        eigenvals = np.linalg.eigvals(result)
        self.assertTrue(np.all(eigenvals > 0))
    
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
    
    def test_cov_sqrtm_single_matrix(self):
        """Test matrix square root."""
        result = cov_sqrtm(self.cov_2x2)
        
        # sqrt(C) @ sqrt(C) should equal C
        reconstructed = result @ result
        np.testing.assert_array_almost_equal(reconstructed, self.cov_2x2, decimal=10)
        
        # Result should be symmetric and positive definite
        np.testing.assert_array_almost_equal(result, result.T, decimal=10)
        eigenvals = np.linalg.eigvals(result)
        self.assertTrue(np.all(eigenvals > 0))
    
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
    
    def test_cov_sqrtm2_single_matrix(self):
        """Test combined square root and reciprocal square root."""
        sqrt_result, rsqrt_result = cov_sqrtm2(self.cov_2x2)
        
        # Compare with individual functions
        expected_sqrt = cov_sqrtm(self.cov_2x2)
        expected_rsqrt = cov_rsqrtm(self.cov_2x2)
        
        np.testing.assert_array_almost_equal(sqrt_result, expected_sqrt, decimal=10)
        np.testing.assert_array_almost_equal(rsqrt_result, expected_rsqrt, decimal=10)
        
        # Verify relationship: sqrt @ rsqrt = identity
        identity_check = sqrt_result @ rsqrt_result
        np.testing.assert_array_almost_equal(identity_check, np.eye(2), decimal=10)
    
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
        self.cov_stack = np.array([
            [[2.0, 0.5], [0.5, 1.5]],
            [[2.2, 0.3], [0.3, 1.8]],
            [[1.8, 0.7], [0.7, 1.2]]
        ])
        
        # Single matrix (should return itself)
        self.single_cov = np.array([[[3.0, 1.0], [1.0, 2.0]]])
    
    def test_single_matrix_mean(self):
        """Test mean of single matrix returns the matrix itself."""
        result = cov_mean(self.single_cov)
        np.testing.assert_array_almost_equal(result, self.single_cov[0], decimal=10)
    
    def test_unweighted_mean(self):
        """Test unweighted mean of covariance matrices."""
        result = cov_mean(self.cov_stack)
        
        # Result should be symmetric and positive definite
        np.testing.assert_array_almost_equal(result, result.T, decimal=10)
        eigenvals = np.linalg.eigvals(result)
        self.assertTrue(np.all(eigenvals > 0))
        
        # Should be roughly in the middle of the input matrices
        expected_trace = np.mean([np.trace(cov) for cov in self.cov_stack])
        actual_trace = np.trace(result)
        self.assertAlmostEqual(actual_trace, expected_trace, delta=0.5)
    
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
    
    def test_robust_mean_geometric_median(self):
        """Test robust mean with geometric median (huber=0)."""
        result = cov_mean(self.cov_stack, robust=True, huber=0)
        
        # Result should be symmetric and positive definite
        np.testing.assert_array_almost_equal(result, result.T, decimal=10)
        eigenvals = np.linalg.eigvals(result)
        self.assertTrue(np.all(eigenvals > 0))
    
    def test_robust_mean_huber(self):
        """Test robust mean with Huber estimator."""
        result = cov_mean(self.cov_stack, robust=True, huber=1.0)
        
        # Result should be symmetric and positive definite
        np.testing.assert_array_almost_equal(result, result.T, decimal=10)
        eigenvals = np.linalg.eigvals(result)
        self.assertTrue(np.all(eigenvals > 0))
    
    def test_convergence_parameters(self):
        """Test convergence parameters (iterations and tolerance)."""
        # Very loose tolerance should converge quickly
        result1 = cov_mean(self.cov_stack, tol=1e-1, iters=5)
        
        # Tight tolerance
        result2 = cov_mean(self.cov_stack, tol=1e-10, iters=100)
        
        # Both should be valid covariance matrices
        for result in [result1, result2]:
            np.testing.assert_array_almost_equal(result, result.T, decimal=10)
            eigenvals = np.linalg.eigvals(result)
            self.assertTrue(np.all(eigenvals > 0))
    
    def test_nancheck_functionality(self):
        """Test NaN checking functionality."""
        # Create data that might cause numerical issues
        problematic_cov = np.array([
            [[1e10, 0], [0, 1e-10]],  # Very different scales
            [[1e-10, 0], [0, 1e10]]
        ])
        
        # Should not raise with nancheck=False (default)
        result = cov_mean(problematic_cov, nancheck=False)
        self.assertFalse(np.any(np.isnan(result)))
        
        # Test that nancheck=True would catch NaNs if they occurred
        # (We can't easily create a case that reliably produces NaNs)
        with patch('numpy.any') as mock_any:
            mock_any.return_value = True  # Simulate NaNs detected
            with self.assertRaises(RuntimeError):
                cov_mean(self.cov_stack, nancheck=True)
    
    def test_verbose_mode(self):
        """Test verbose mode output."""
        with patch('eegprep.utils.covariance.logger.info') as mock_log:
            cov_mean(self.cov_stack, robust=True, huber=None, verbose=True)
            # Should have logged median deviations
            mock_log.assert_called()


class TestCovShrinkage(unittest.TestCase):
    """Test the covariance shrinkage function."""
    
    def setUp(self):
        """Create test covariance matrices."""
        self.cov_2x2 = np.array([[4.0, 2.0], [2.0, 3.0]])
        self.cov_3x3 = np.array([
            [5.0, 1.0, 0.5],
            [1.0, 4.0, 1.5],
            [0.5, 1.5, 3.0]
        ])
        
        # Stack of matrices
        self.cov_stack = np.array([
            [[4.0, 2.0], [2.0, 3.0]],
            [[6.0, 1.0], [1.0, 2.0]]
        ])
    
    def test_no_shrinkage(self):
        """Test that zero shrinkage returns original matrix."""
        result = cov_shrinkage(self.cov_2x2, shrinkage=0)
        np.testing.assert_array_equal(result, self.cov_2x2)
        
        # Test with different targets (should all return original)
        for target in ['eye', 'scaled-eye', 'diag']:
            result = cov_shrinkage(self.cov_2x2, shrinkage=0, target=target)
            np.testing.assert_array_equal(result, self.cov_2x2)
    
    def test_full_shrinkage_eye(self):
        """Test full shrinkage towards identity."""
        result = cov_shrinkage(self.cov_2x2, shrinkage=1.0, target='eye')
        expected = np.eye(2)
        np.testing.assert_array_equal(result, expected)
    
    def test_full_shrinkage_scaled_eye(self):
        """Test full shrinkage towards scaled identity."""
        result = cov_shrinkage(self.cov_2x2, shrinkage=1.0, target='scaled-eye')
        
        # Should be scaled identity with scale = trace/N
        trace = np.trace(self.cov_2x2)
        scale = trace / 2
        expected = scale * np.eye(2)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_full_shrinkage_diag(self):
        """Test full shrinkage towards diagonal."""
        result = cov_shrinkage(self.cov_2x2, shrinkage=1.0, target='diag')
        expected = np.diag(np.diag(self.cov_2x2))
        np.testing.assert_array_equal(result, expected)
    
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
    
    def test_stack_shrinkage(self):
        """Test shrinkage on stack of matrices."""
        shrinkage = 0.5
        result = cov_shrinkage(self.cov_stack, shrinkage=shrinkage, target='eye')
        
        # Check shape preserved
        self.assertEqual(result.shape, self.cov_stack.shape)
        
        # Check each matrix individually
        for i in range(len(self.cov_stack)):
            expected = shrinkage * np.eye(2) + (1 - shrinkage) * self.cov_stack[i]
            np.testing.assert_array_almost_equal(result[i], expected)
    
    def test_3d_matrices_shrinkage(self):
        """Test shrinkage with 3D matrices."""
        result = cov_shrinkage(self.cov_3x3, shrinkage=0.4, target='diag')
        
        # Should preserve diagonal elements and shrink off-diagonal
        np.testing.assert_array_almost_equal(np.diag(result), np.diag(self.cov_3x3))
        
        # Off-diagonal should be shrunk
        off_diag_orig = self.cov_3x3[0, 1]
        off_diag_result = result[0, 1]
        self.assertLess(abs(off_diag_result), abs(off_diag_orig))
    
    def test_scaled_eye_with_stack(self):
        """Test scaled-eye target with stack of matrices."""
        result = cov_shrinkage(self.cov_stack, shrinkage=1.0, target='scaled-eye')
        
        # Each matrix should be scaled identity
        for i in range(len(self.cov_stack)):
            trace = np.trace(self.cov_stack[i])
            scale = trace / 2
            expected = scale * np.eye(2)
            np.testing.assert_array_almost_equal(result[i], expected)
    
    def test_invalid_target_error(self):
        """Test error for invalid shrinkage target."""
        with self.assertRaises(ValueError) as cm:
            cov_shrinkage(self.cov_2x2, shrinkage=0.5, target='invalid')
        self.assertIn('Unsupported shrinkage target', str(cm.exception))
    
    def test_symmetric_output(self):
        """Test that output maintains symmetry."""
        for target in ['eye', 'scaled-eye', 'diag']:
            result = cov_shrinkage(self.cov_2x2, shrinkage=0.3, target=target)
            np.testing.assert_array_almost_equal(result, result.T, decimal=10)
    
    def test_positive_definite_preservation(self):
        """Test that positive definiteness is preserved."""
        for target in ['eye', 'scaled-eye', 'diag']:
            result = cov_shrinkage(self.cov_2x2, shrinkage=0.7, target=target)
            eigenvals = np.linalg.eigvals(result)
            self.assertTrue(np.all(eigenvals > 0))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and numerical stability."""
    
    def test_near_singular_matrices(self):
        """Test operations on near-singular matrices."""
        # Create a matrix with very small eigenvalues
        near_singular = np.array([[1.0, 0.999], [0.999, 1.0]])
        
        # All operations should still work
        log_result = cov_logm(near_singular)
        exp_result = cov_expm(log_result)
        sqrt_result = cov_sqrtm(near_singular)
        rsqrt_result = cov_rsqrtm(near_singular)
        
        # Check round-trip accuracy
        np.testing.assert_array_almost_equal(exp_result, near_singular, decimal=8)
        reconstructed = sqrt_result @ sqrt_result
        np.testing.assert_array_almost_equal(reconstructed, near_singular, decimal=8)
    
    def test_identity_matrix_operations(self):
        """Test operations on identity matrix."""
        identity = np.eye(3)
        
        # Log of identity should be zero matrix
        log_result = cov_logm(identity)
        np.testing.assert_array_almost_equal(log_result, np.zeros((3, 3)), decimal=10)
        
        # Square root of identity should be identity
        sqrt_result = cov_sqrtm(identity)
        np.testing.assert_array_almost_equal(sqrt_result, identity, decimal=10)
        
        # Reciprocal square root of identity should be identity
        rsqrt_result = cov_rsqrtm(identity)
        np.testing.assert_array_almost_equal(rsqrt_result, identity, decimal=10)
    
    def test_large_matrices(self):
        """Test operations on larger matrices."""
        # Create a 10x10 positive definite matrix
        np.random.seed(42)  # For reproducibility
        A = np.random.randn(10, 10)
        large_cov = A @ A.T + np.eye(10)  # Ensure positive definite
        
        # Test basic operations
        sqrt_result = cov_sqrtm(large_cov)
        reconstructed = sqrt_result @ sqrt_result
        np.testing.assert_array_almost_equal(reconstructed, large_cov, decimal=8)
        
        # Test shrinkage
        shrunk = cov_shrinkage(large_cov, shrinkage=0.1, target='eye')
        eigenvals = np.linalg.eigvals(shrunk)
        self.assertTrue(np.all(eigenvals > 0))
    
    def test_numerical_precision_consistency(self):
        """Test that operations maintain numerical precision."""
        # Use different dtypes
        cov_float32 = self.cov_2x2.astype(np.float32)
        cov_float64 = self.cov_2x2.astype(np.float64)
        
        # Results should be consistent (within precision limits)
        sqrt32 = cov_sqrtm(cov_float32)
        sqrt64 = cov_sqrtm(cov_float64)
        
        # Float32 should be close to float64 (within single precision)
        np.testing.assert_array_almost_equal(sqrt32, sqrt64.astype(np.float32), decimal=6)
    
    def setUp(self):
        """Set up test matrices."""
        self.cov_2x2 = np.array([[2.0, 1.0], [1.0, 2.0]])


if __name__ == '__main__':
    unittest.main()
