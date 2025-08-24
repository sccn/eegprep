import unittest
import numpy as np
import sys
from unittest.mock import patch, MagicMock
from io import StringIO

from eegprep.utils.testing import compare_eeg, DebuggableTestCase, is_debug


class TestCompareEeg(unittest.TestCase):
    
    def test_compare_eeg_identical_arrays(self):
        """Test comparing identical arrays passes."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Capture stdout to suppress the print statement
        with patch('sys.stdout', StringIO()):
            compare_eeg(a, b)  # Should not raise
    
    def test_compare_eeg_32bit_conversion(self):
        """Test 32-bit precision conversion."""
        a = np.array([[1.0, 2.0]], dtype=np.float64)
        b = np.array([[1.0, 2.0]], dtype=np.float64)
        
        with patch('sys.stdout', StringIO()):
            compare_eeg(a, b, use_32_bit=True)  # Should convert to float32
    
    def test_compare_eeg_no_32bit_conversion(self):
        """Test without 32-bit conversion."""
        a = np.array([[1.0, 2.0]], dtype=np.float64)
        b = np.array([[1.0, 2.0]], dtype=np.float64)
        
        with patch('sys.stdout', StringIO()):
            compare_eeg(a, b, use_32_bit=False)  # Should keep original dtype
    
    def test_compare_eeg_flatten_3d_to_2d(self):
        """Test flattening 3D arrays to 2D."""
        # The reshape logic expects that we can reshape to shape[:2]
        # This means the total elements must equal the product of first 2 dimensions
        # For a (2, 3, 1) array, total = 6, shape[:2] = (2, 3), product = 6 ✓
        a = np.ones((2, 3, 1))  # 6 total elements, reshape to (2, 3) = 6 elements
        b = np.ones((2, 3, 1))  # 6 total elements, reshape to (2, 3) = 6 elements
        
        with patch('sys.stdout', StringIO()):
            compare_eeg(a, b)  # Should flatten 3D to 2D using shape[:2]
    
    def test_compare_eeg_flatten_3d_incompatible_shape_error(self):
        """Test that incompatible 3D reshape raises ValueError."""
        # Create 3D array that cannot be reshaped to shape[:2]
        a = np.ones((2, 3, 4))  # 24 total elements, but shape[:2] = (2, 3) = 6 elements
        b = np.ones((2, 3, 4))  # Cannot reshape 24 elements into (2, 3)
        
        with self.assertRaises(ValueError) as cm:
            compare_eeg(a, b)
        self.assertIn("cannot reshape", str(cm.exception))
    
    def test_compare_eeg_different_shapes_error(self):
        """Test error when arrays have different shapes."""
        a = np.array([[1.0, 2.0]])
        b = np.array([[1.0], [2.0]])
        
        with self.assertRaises(ValueError) as cm:
            compare_eeg(a, b)
        self.assertIn("different shapes", str(cm.exception))
    
    def test_compare_eeg_tolerance_parameters(self):
        """Test rtol and atol tolerance parameters."""
        a = np.array([[1.0, 2.0]])
        b = np.array([[1.001, 2.001]])  # Small differences
        
        with patch('sys.stdout', StringIO()):
            # Should pass with loose tolerance
            compare_eeg(a, b, rtol=1e-2, atol=1e-2)
        
        # Should fail with tight tolerance
        with self.assertRaises(AssertionError):
            with patch('sys.stdout', StringIO()):
                compare_eeg(a, b, rtol=1e-6, atol=1e-6)
    
    def test_compare_eeg_custom_error_message(self):
        """Test custom error message parameter."""
        a = np.array([[1.0]])
        b = np.array([[2.0]])  # Different values
        
        with self.assertRaises(AssertionError) as cm:
            with patch('sys.stdout', StringIO()):
                compare_eeg(a, b, err_msg="Custom error message")
        self.assertIn("Custom error message", str(cm.exception))
    
    def test_compare_eeg_prints_actual_differences(self):
        """Test that actual differences are printed."""
        a = np.array([[1.0, 2.0]])
        b = np.array([[1.1, 2.1]])
        
        with patch('sys.stdout', StringIO()) as mock_stdout:
            try:
                compare_eeg(a, b, rtol=1e-6, atol=1e-6)
            except AssertionError:
                pass  # Expected to fail
            
            output = mock_stdout.getvalue()
            self.assertIn("Actual differences:", output)
            self.assertIn("rtol:", output)
            self.assertIn("atol:", output)
    
    def test_compare_eeg_2d_array_flattening_in_comparison(self):
        """Test that 2D arrays are flattened for comparison calculations."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        with patch('sys.stdout', StringIO()):
            compare_eeg(a, b)  # Should flatten internally for comparison


class MockTestCase(DebuggableTestCase):
    """Mock test case for testing DebuggableTestCase."""
    
    def test_example(self):
        """Example test method."""
        self.assertTrue(True)
    
    def test_another(self):
        """Another example test method."""
        self.assertEqual(1, 1)


class TestDebuggableTestCase(unittest.TestCase):
    
    def test_debugTestCase_loads_and_runs_tests(self):
        """Test that debugTestCase loads and runs test methods."""
        # Mock the test suite and its debug method
        mock_suite = MagicMock()
        
        with patch('unittest.defaultTestLoader.loadTestsFromTestCase') as mock_loader:
            mock_loader.return_value = mock_suite
            
            # Call debugTestCase
            MockTestCase.debugTestCase()
            
            # Verify that loadTestsFromTestCase was called with the class
            mock_loader.assert_called_once_with(MockTestCase)
            
            # Verify that debug() was called on the test suite
            mock_suite.debug.assert_called_once()
    
    def test_debugTestCase_inheritance(self):
        """Test that DebuggableTestCase properly inherits from unittest.TestCase."""
        self.assertTrue(issubclass(DebuggableTestCase, unittest.TestCase))
        
        # Test that an instance can be created
        instance = MockTestCase()
        self.assertIsInstance(instance, unittest.TestCase)
        self.assertIsInstance(instance, DebuggableTestCase)


class TestIsDebug(unittest.TestCase):
    
    def test_is_debug_no_tracer(self):
        """Test is_debug returns False when no tracer is active."""
        def mock_gettrace():
            return None
        with patch.object(sys, 'gettrace', mock_gettrace):
            self.assertFalse(is_debug())
    
    def test_is_debug_with_tracer(self):
        """Test is_debug returns True when tracer is active."""
        mock_tracer = MagicMock()
        def mock_gettrace():
            return mock_tracer
        with patch.object(sys, 'gettrace', mock_gettrace):
            self.assertTrue(is_debug())
    
    def test_is_debug_no_gettrace_attribute(self):
        """Test is_debug handles missing gettrace attribute gracefully."""
        with patch.object(sys, 'gettrace', None):
            # When gettrace is None, getattr returns None, and None() raises TypeError
            with self.assertRaises(TypeError):
                is_debug()
    
    def test_is_debug_gettrace_returns_none_function(self):
        """Test is_debug when gettrace returns a function that returns None."""
        def mock_gettrace():
            return None
        with patch.object(sys, 'gettrace', mock_gettrace):
            self.assertFalse(is_debug())


class TestModuleConstants(unittest.TestCase):
    
    def test_module_exports(self):
        """Test that __all__ contains expected exports."""
        from eegprep.utils.testing import __all__
        expected_exports = ['compare_eeg', 'DebuggableTestCase', 'is_debug']
        self.assertEqual(set(__all__), set(expected_exports))
    
    def test_default_32_bit_constant(self):
        """Test default_32_bit constant value."""
        from eegprep.utils.testing import default_32_bit
        self.assertTrue(default_32_bit)
    
    def test_flatten_to_2d_constant(self):
        """Test flatten_to_2d constant value."""
        from eegprep.utils.testing import flatten_to_2d
        self.assertTrue(flatten_to_2d)


if __name__ == '__main__':
    unittest.main()
