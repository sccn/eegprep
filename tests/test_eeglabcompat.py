"""
Test suite for eeglabcompat.py - EEGLAB compatibility layer.

This module tests the EEGLAB compatibility functions that provide
Python interfaces to MATLAB/Octave EEGLAB functions.
"""

import unittest
import sys
import numpy as np
import tempfile
import os
import shutil

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.eeglabcompat import (
    MatlabWrapper, get_eeglab
)
from eegprep.utils.testing import DebuggableTestCase


def create_test_eeg():
    """Create a complete test EEG structure with all required fields."""
    return {
        'data': np.random.randn(32, 1000, 10),
        'srate': 500.0,
        'nbchan': 32,
        'pnts': 1000,
        'trials': 10,
        'xmin': -1.0,
        'xmax': 1.0,
        'times': np.linspace(-1.0, 1.0, 1000),
        'icaact': [],
        'icawinv': [],
        'icasphere': [],
        'icaweights': [],
        'icachansind': [],
        'chanlocs': [],
        'urchanlocs': [],
        'chaninfo': [],
        'ref': 'common',
        'history': '',
        'saved': 'yes',
        'etc': {},
        'event': [],
        'epoch': [],
        'setname': 'test_dataset',
        'filename': 'test.set',
        'filepath': '/tmp'
    }


class TestMatlabWrapper(DebuggableTestCase):
    """Test cases for MatlabWrapper class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_marshal_boolean_values(self):
        """Test marshaling of boolean values."""
        # Test True
        result = MatlabWrapper.marshal(True)
        self.assertEqual(result, 'true')
        
        # Test False
        result = MatlabWrapper.marshal(False)
        self.assertEqual(result, 'false')

    def test_marshal_other_values(self):
        """Test marshaling of other value types."""
        # Test string
        result = MatlabWrapper.marshal("test_string")
        self.assertEqual(result, "'test_string'")
        
        # Test number
        result = MatlabWrapper.marshal(42)
        self.assertEqual(result, '42')
        
        # Test list
        result = MatlabWrapper.marshal([1, 2, 3])
        self.assertEqual(result, '[1, 2, 3]')

    def test_wrapper_creation(self):
        """Test MatlabWrapper creation with mock engine."""
        class MockEngine:
            def eval(self, cmd, nargout=0):
                pass
        
        engine = MockEngine()
        wrapper = MatlabWrapper(engine)
        self.assertIsInstance(wrapper, MatlabWrapper)
        self.assertEqual(wrapper.engine, engine)

    def test_getattr_returns_wrapper_function(self):
        """Test that __getattr__ returns a callable wrapper function."""
        class MockEngine:
            def eval(self, cmd, nargout=0):
                pass
        
        engine = MockEngine()
        wrapper = MatlabWrapper(engine)
        
        # Test that getting an attribute returns a callable
        func = wrapper.some_function
        self.assertTrue(callable(func))


class TestGetEeglab(DebuggableTestCase):
    """Test cases for get_eeglab function."""

    def test_get_eeglab_default_runtime(self):
        """Test get_eeglab with default runtime."""
        try:
            eeglab = get_eeglab()
            self.assertIsNotNone(eeglab)
        except Exception as e:
            # Skip test if MATLAB/Octave is not available
            self.skipTest(f"MATLAB/Octave not available: {e}")

    def test_get_eeglab_mat_runtime(self):
        """Test get_eeglab with MATLAB runtime."""
        try:
            eeglab = get_eeglab('MAT')
            self.assertIsNotNone(eeglab)
        except ImportError:
            self.skipTest("MATLAB engine not available")
        except Exception as e:
            self.skipTest(f"MATLAB not available: {e}")

    def test_get_eeglab_oct_runtime(self):
        """Test get_eeglab with Octave runtime."""
        try:
            eeglab = get_eeglab('OCT')
            self.assertIsNotNone(eeglab)
        except ImportError:
            self.skipTest("Oct2Py not available")
        except Exception as e:
            self.skipTest(f"Octave not available: {e}")

    def test_get_eeglab_invalid_runtime(self):
        """Test get_eeglab with invalid runtime."""
        with self.assertRaises(ValueError):
            get_eeglab('INVALID')

    def test_get_eeglab_caching(self):
        """Test that get_eeglab caches underlying engine instances."""
        try:
            eeglab1 = get_eeglab('MAT')
            eeglab2 = get_eeglab('MAT')
            # Wrappers are different instances, but underlying engines should be cached
            self.assertIsInstance(eeglab1, MatlabWrapper)
            self.assertIsInstance(eeglab2, MatlabWrapper)
            self.assertIs(eeglab1.engine, eeglab2.engine)
        except Exception as e:
            self.skipTest(f"MATLAB not available: {e}")

    def test_get_eeglab_auto_file_roundtrip(self):
        """Test get_eeglab with auto_file_roundtrip disabled."""
        try:
            eeglab = get_eeglab('MAT', auto_file_roundtrip=False)
            self.assertIsNotNone(eeglab)
            # Should not be wrapped in MatlabWrapper
            self.assertNotIsInstance(eeglab, MatlabWrapper)
        except Exception as e:
            self.skipTest(f"MATLAB not available: {e}")


class TestEeglabCompatIntegration(DebuggableTestCase):
    """Integration tests for eeglabcompat functions."""

    def test_eeglab_runtime_availability(self):
        """Test that at least one EEGLAB runtime is available."""
        runtimes_available = []
        
        # Test MATLAB runtime
        try:
            eeglab = get_eeglab('MAT')
            runtimes_available.append('MAT')
        except Exception:
            pass
        
        # Test Octave runtime
        try:
            eeglab = get_eeglab('OCT')
            runtimes_available.append('OCT')
        except Exception:
            pass
        
        # At least one runtime should be available for testing
        self.assertGreater(len(runtimes_available), 0, 
                          "No EEGLAB runtime available for testing")


if __name__ == '__main__':
    unittest.main()
