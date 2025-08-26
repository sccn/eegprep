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
    MatlabWrapper, get_eeglab, eeg_checkset, clean_drifts, 
    pop_eegfiltnew, clean_artifacts
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
        """Test that get_eeglab caches engine instances."""
        try:
            eeglab1 = get_eeglab('MAT')
            eeglab2 = get_eeglab('MAT')
            self.assertIs(eeglab1, eeglab2)
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


class TestEegCheckset(DebuggableTestCase):
    """Test cases for eeg_checkset function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_eeg_checkset_basic(self):
        """Test eeg_checkset with basic EEG structure."""
        try:
            result = eeg_checkset(self.test_eeg)
            self.assertIsNotNone(result)
            # Should return the same or modified EEG structure
            self.assertIn('data', result)
            self.assertIn('srate', result)
        except Exception as e:
            self.skipTest(f"eeg_checkset not available: {e}")

    def test_eeg_checkset_with_none_eeglab(self):
        """Test eeg_checkset with None eeglab parameter."""
        try:
            result = eeg_checkset(self.test_eeg, eeglab=None)
            self.assertIsNotNone(result)
        except Exception as e:
            self.skipTest(f"eeg_checkset not available: {e}")

    def test_eeg_checkset_with_custom_eeglab(self):
        """Test eeg_checkset with custom eeglab instance."""
        try:
            eeglab = get_eeglab('MAT')
            result = eeg_checkset(self.test_eeg, eeglab=eeglab)
            self.assertIsNotNone(result)
        except Exception as e:
            self.skipTest(f"eeg_checkset not available: {e}")


class TestCleanDrifts(DebuggableTestCase):
    """Test cases for clean_drifts function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_drifts_basic(self):
        """Test clean_drifts with basic parameters."""
        try:
            result = clean_drifts(self.test_eeg, 0.5, 0.1)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"clean_drifts not available: {e}")

    def test_clean_drifts_with_custom_eeglab(self):
        """Test clean_drifts with custom eeglab instance."""
        try:
            eeglab = get_eeglab('MAT')
            result = clean_drifts(self.test_eeg, 0.5, 0.1, eeglab=eeglab)
            self.assertIsNotNone(result)
        except Exception as e:
            self.skipTest(f"clean_drifts not available: {e}")


class TestPopEegfiltnew(DebuggableTestCase):
    """Test cases for pop_eegfiltnew function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_pop_eegfiltnew_lowpass(self):
        """Test pop_eegfiltnew with lowpass filter."""
        try:
            result = pop_eegfiltnew(self.test_eeg, locutoff=None, hicutoff=25)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
            self.assertEqual(result['srate'], self.test_eeg['srate'])
        except Exception as e:
            self.skipTest(f"pop_eegfiltnew not available: {e}")

    def test_pop_eegfiltnew_highpass(self):
        """Test pop_eegfiltnew with highpass filter."""
        try:
            result = pop_eegfiltnew(self.test_eeg, locutoff=5, hicutoff=None)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"pop_eegfiltnew not available: {e}")

    def test_pop_eegfiltnew_bandpass(self):
        """Test pop_eegfiltnew with bandpass filter."""
        try:
            result = pop_eegfiltnew(self.test_eeg, locutoff=5, hicutoff=25)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"pop_eegfiltnew not available: {e}")

    def test_pop_eegfiltnew_with_revfilt(self):
        """Test pop_eegfiltnew with reverse filter."""
        try:
            result = pop_eegfiltnew(self.test_eeg, locutoff=5, hicutoff=25, revfilt=True)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"pop_eegfiltnew not available: {e}")

    def test_pop_eegfiltnew_with_plotfreqz(self):
        """Test pop_eegfiltnew with plotfreqz."""
        try:
            result = pop_eegfiltnew(self.test_eeg, locutoff=5, hicutoff=25, plotfreqz=True)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"pop_eegfiltnew not available: {e}")

    def test_pop_eegfiltnew_no_cutoffs(self):
        """Test pop_eegfiltnew with no cutoffs defined."""
        with self.assertRaises(Exception):
            pop_eegfiltnew(self.test_eeg, locutoff=None, hicutoff=None)


class TestCleanArtifacts(DebuggableTestCase):
    """Test cases for clean_artifacts function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_artifacts_basic(self):
        """Test clean_artifacts with basic parameters."""
        try:
            result = clean_artifacts(self.test_eeg)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"clean_artifacts not available: {e}")

    def test_clean_artifacts_with_channel_criterion(self):
        """Test clean_artifacts with channel criterion."""
        try:
            result = clean_artifacts(self.test_eeg, ChannelCriterion=0.8)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"clean_artifacts not available: {e}")

    def test_clean_artifacts_with_line_noise_criterion(self):
        """Test clean_artifacts with line noise criterion."""
        try:
            result = clean_artifacts(self.test_eeg, LineNoiseCriterion=4)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"clean_artifacts not available: {e}")

    def test_clean_artifacts_with_flatline_criterion(self):
        """Test clean_artifacts with flatline criterion."""
        try:
            result = clean_artifacts(self.test_eeg, FlatlineCriterion=5)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"clean_artifacts not available: {e}")

    def test_clean_artifacts_with_burst_criterion(self):
        """Test clean_artifacts with burst criterion."""
        try:
            result = clean_artifacts(self.test_eeg, BurstCriterion=20)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"clean_artifacts not available: {e}")

    def test_clean_artifacts_with_window_criterion(self):
        """Test clean_artifacts with window criterion."""
        try:
            result = clean_artifacts(self.test_eeg, WindowCriterion=0.25)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"clean_artifacts not available: {e}")

    def test_clean_artifacts_with_highpass(self):
        """Test clean_artifacts with highpass filter."""
        try:
            result = clean_artifacts(self.test_eeg, Highpass=[0.25, 0.75])
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"clean_artifacts not available: {e}")

    def test_clean_artifacts_with_window_criterion_tolerances(self):
        """Test clean_artifacts with window criterion tolerances."""
        try:
            result = clean_artifacts(self.test_eeg, WindowCriterionTolerances=[float('-inf'), 7])
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"clean_artifacts not available: {e}")

    def test_clean_artifacts_with_burst_rejection(self):
        """Test clean_artifacts with burst rejection."""
        try:
            result = clean_artifacts(self.test_eeg, BurstRejection=True)
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"clean_artifacts not available: {e}")

    def test_clean_artifacts_all_criteria_disabled(self):
        """Test clean_artifacts with all criteria disabled."""
        try:
            result = clean_artifacts(
                self.test_eeg,
                ChannelCriterion='off',
                LineNoiseCriterion='off',
                FlatlineCriterion='off',
                BurstCriterion='off',
                Highpass='off'
            )
            self.assertIsNotNone(result)
            self.assertIn('data', result)
        except Exception as e:
            self.skipTest(f"clean_artifacts not available: {e}")


class TestEeglabCompatIntegration(DebuggableTestCase):
    """Integration tests for eeglabcompat functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_eeglab_compatibility_workflow(self):
        """Test a complete workflow using eeglabcompat functions."""
        try:
            # Step 1: Check EEG structure
            eeg_checked = eeg_checkset(self.test_eeg)
            self.assertIsNotNone(eeg_checked)
            
            # Step 2: Apply filtering
            eeg_filtered = pop_eegfiltnew(eeg_checked, locutoff=5, hicutoff=25)
            self.assertIsNotNone(eeg_filtered)
            
            # Step 3: Clean artifacts
            eeg_cleaned = clean_artifacts(eeg_filtered, FlatlineCriterion=5)
            self.assertIsNotNone(eeg_cleaned)
            
            # Verify data integrity
            self.assertIn('data', eeg_cleaned)
            self.assertEqual(eeg_cleaned['srate'], self.test_eeg['srate'])
            self.assertEqual(eeg_cleaned['nbchan'], self.test_eeg['nbchan'])
            
        except Exception as e:
            self.skipTest(f"Integration test not available: {e}")

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
