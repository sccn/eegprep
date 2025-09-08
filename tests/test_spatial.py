"""
Tests for eegprep.utils.spatial module.

This module tests the spherical spline interpolation functions used in topographic plotting.
"""

import unittest
import numpy as np
from eegprep.utils.spatial import sphericalSplineInterpolate, _interpMx


class TestInterpMx(unittest.TestCase):
    """Test the _interpMx helper function for Legendre polynomial approximation."""
    
    def test_interp_mx_basic(self):
        """Test basic _interpMx functionality."""
        # Test with simple cosine values
        cosang = np.array([[1.0, 0.5, 0.0, -0.5, -1.0]])
        G, H = _interpMx(cosang, order=4, tol=1e-10)
        
        # Should return matrices with same shape as input
        self.assertEqual(G.shape, cosang.shape)
        self.assertEqual(H.shape, cosang.shape)
        
        # Should be finite
        self.assertTrue(np.all(np.isfinite(G)))
        self.assertTrue(np.all(np.isfinite(H)))
    
    def test_interp_mx_single_point(self):
        """Test _interpMx with single point."""
        cosang = np.array([[1.0]])
        G, H = _interpMx(cosang, order=4, tol=1e-10)
        
        self.assertEqual(G.shape, (1, 1))
        self.assertEqual(H.shape, (1, 1))
        self.assertTrue(np.isfinite(G[0, 0]))
        self.assertTrue(np.isfinite(H[0, 0]))
    
    def test_interp_mx_edge_values(self):
        """Test _interpMx with edge cosine values."""
        # Test extreme cosine values
        cosang = np.array([[-1.0, -0.999, 0.0, 0.999, 1.0]])
        G, H = _interpMx(cosang, order=4, tol=1e-10)
        
        # All values should be finite
        self.assertTrue(np.all(np.isfinite(G)))
        self.assertTrue(np.all(np.isfinite(H)))
    
    def test_interp_mx_numerical_stability(self):
        """Test _interpMx numerical stability with various inputs."""
        # Test with very small differences
        cosang = np.array([[1.0, 0.9999, 0.9998, 0.9997]])
        G, H = _interpMx(cosang, order=4, tol=1e-10)
        
        self.assertTrue(np.all(np.isfinite(G)))
        self.assertTrue(np.all(np.isfinite(H)))
    
    def test_interp_mx_empty_input(self):
        """Test _interpMx with empty input."""
        cosang = np.array([[]])
        G, H = _interpMx(cosang, order=4, tol=1e-10)
        
        self.assertEqual(G.size, 0)
        self.assertEqual(H.size, 0)


class TestSphericalSplineInterpolate(unittest.TestCase):
    """Test the sphericalSplineInterpolate function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple 3D electrode positions (as expected by the function)
        # 4 electrodes on unit sphere
        self.src_positions = np.array([
            [1.0, 0.0, 0.0, -1.0],    # x coordinates
            [0.0, 1.0, 0.0, 0.0],     # y coordinates  
            [0.0, 0.0, 1.0, 0.0]      # z coordinates
        ])
        
        # Destination positions (where to interpolate)
        self.dest_positions = np.array([
            [0.707, -0.707],    # x coordinates
            [0.707, 0.707],     # y coordinates
            [0.0, 0.0]          # z coordinates
        ])
    
    def test_spherical_spline_interpolate_basic(self):
        """Test basic sphericalSplineInterpolate functionality."""
        W, Gss, Gds, Hds = sphericalSplineInterpolate(
            self.src_positions, 
            self.dest_positions
        )
        
        # Check return value shapes
        n_src = self.src_positions.shape[1]
        n_dest = self.dest_positions.shape[1]
        
        self.assertEqual(W.shape, (n_dest, n_src))
        self.assertEqual(Gss.shape, (n_src, n_src))
        self.assertEqual(Gds.shape, (n_dest, n_src))
        self.assertEqual(Hds.shape, (n_dest, n_src))
        
        # All values should be finite
        self.assertTrue(np.all(np.isfinite(W)))
        self.assertTrue(np.all(np.isfinite(Gss)))
        self.assertTrue(np.all(np.isfinite(Gds)))
        self.assertTrue(np.all(np.isfinite(Hds)))
    
    def test_spherical_spline_interpolate_different_types(self):
        """Test sphericalSplineInterpolate with different interpolation types."""
        # Test 'spline' type (default)
        W_spline, _, _, _ = sphericalSplineInterpolate(
            self.src_positions, 
            self.dest_positions,
            type='spline'
        )
        
        # Test 'slap' type
        W_slap, _, _, _ = sphericalSplineInterpolate(
            self.src_positions, 
            self.dest_positions,
            type='slap'
        )
        
        # Both should return finite matrices of the same shape
        self.assertEqual(W_spline.shape, W_slap.shape)
        self.assertTrue(np.all(np.isfinite(W_spline)))
        self.assertTrue(np.all(np.isfinite(W_slap)))
        
        # Results should be different for different interpolation types
        self.assertFalse(np.allclose(W_spline, W_slap))
    
    def test_spherical_spline_interpolate_regularization(self):
        """Test sphericalSplineInterpolate with different regularization parameters."""
        # Test with different lambda values
        W_low_reg, _, _, _ = sphericalSplineInterpolate(
            self.src_positions, 
            self.dest_positions,
            lambda_reg=1e-10
        )
        
        W_high_reg, _, _, _ = sphericalSplineInterpolate(
            self.src_positions, 
            self.dest_positions,
            lambda_reg=1e-2
        )
        
        # Both should be finite and same shape
        self.assertEqual(W_low_reg.shape, W_high_reg.shape)
        self.assertTrue(np.all(np.isfinite(W_low_reg)))
        self.assertTrue(np.all(np.isfinite(W_high_reg)))
    
    def test_spherical_spline_interpolate_different_orders(self):
        """Test sphericalSplineInterpolate with different polynomial orders."""
        # Test with different orders
        W_order2, _, _, _ = sphericalSplineInterpolate(
            self.src_positions, 
            self.dest_positions,
            order=2
        )
        
        W_order6, _, _, _ = sphericalSplineInterpolate(
            self.src_positions, 
            self.dest_positions,
            order=6
        )
        
        # Both should be finite and same shape
        self.assertEqual(W_order2.shape, W_order6.shape)
        self.assertTrue(np.all(np.isfinite(W_order2)))
        self.assertTrue(np.all(np.isfinite(W_order6)))


class TestSphericalSplineInterpolateErrorHandling(unittest.TestCase):
    """Test error handling in sphericalSplineInterpolate."""
    
    def test_invalid_input_dimensions(self):
        """Test error handling for invalid input dimensions."""
        # Wrong shape for src (should be 3 x N)
        src_wrong = np.array([[1, 2], [3, 4]])  # 2x2 instead of 3xN
        dest = np.array([[0.5], [0.5], [0.0]])  # 3x1
        
        with self.assertRaises(ValueError):
            sphericalSplineInterpolate(src_wrong, dest)
    
    def test_invalid_interpolation_type(self):
        """Test error handling for invalid interpolation type."""
        src = np.array([[1, 0], [0, 1], [0, 0]])  # 3x2
        dest = np.array([[0.5], [0.5], [0.0]])   # 3x1
        
        with self.assertRaises(ValueError):
            sphericalSplineInterpolate(src, dest, type='invalid_type')
    
    def test_empty_inputs(self):
        """Test error handling for empty inputs."""
        empty_src = np.array([[], [], []])  # 3x0
        dest = np.array([[0.5], [0.5], [0.0]])  # 3x1
        
        # This should either raise an error or handle gracefully
        try:
            W, _, _, _ = sphericalSplineInterpolate(empty_src, dest)
            # If it succeeds, W should be empty
            self.assertEqual(W.shape[1], 0)
        except (ValueError, np.linalg.LinAlgError):
            # Expected behavior for empty input
            pass


if __name__ == '__main__':
    unittest.main()

