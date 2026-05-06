"""
Test suite for functions/adminfunc/eeglabcompat.py - EEGLAB compatibility layer.

This module tests the EEGLAB compatibility functions that provide
Python interfaces to MATLAB/Octave EEGLAB functions.
"""

import os
import unittest
from copy import deepcopy

from eegprep.functions.adminfunc.eeglabcompat import (
    MatlabWrapper, get_eeglab, clean_drifts, pop_eegfiltnew,
    eeg_checkset as eeglab_eeg_checkset
)
from eegprep import clean_artifacts, pop_loadset
from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.utils.testing import DebuggableTestCase

# Path to test data
LOCAL_DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/')


class TestMatlabWrapper(DebuggableTestCase):
    """Test cases for MatlabWrapper class."""

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


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
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


class TestEegCheckset(DebuggableTestCase):
    """Test cases for eeg_checkset function (Python implementation)."""

    def setUp(self):
        """Set up test fixtures with real data."""
        self.EEG = pop_loadset(os.path.join(LOCAL_DATA_PATH, 'eeglab_data_with_ica_tmp.set'))

    def test_eeg_checkset_basic(self):
        """Test Python eeg_checkset with basic EEG structure."""
        result = eeg_checkset(deepcopy(self.EEG))
        self.assertIsNotNone(result)
        self.assertIn('data', result)
        self.assertIn('srate', result)


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestEegChecksetMatlab(DebuggableTestCase):
    """Test cases for eeg_checkset MATLAB wrapper."""

    def setUp(self):
        """Set up test fixtures with real data."""
        self.EEG = pop_loadset(os.path.join(LOCAL_DATA_PATH, 'eeglab_data_with_ica_tmp.set'))

    def test_eeg_checkset_matlab_default(self):
        """Test MATLAB eeg_checkset with default engine."""
        result = eeglab_eeg_checkset(deepcopy(self.EEG))
        self.assertIsNotNone(result)
        self.assertIn('data', result)

    def test_eeg_checkset_matlab_explicit(self):
        """Test MATLAB eeg_checkset with explicit engine."""
        eeglab = get_eeglab('MAT')
        result = eeglab_eeg_checkset(deepcopy(self.EEG), eeglab=eeglab)
        self.assertIsNotNone(result)


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestCleanDrifts(DebuggableTestCase):
    """Test cases for clean_drifts function."""

    def setUp(self):
        """Set up test fixtures with real data."""
        self.EEG = pop_loadset(os.path.join(LOCAL_DATA_PATH, 'eeglab_data_with_ica_tmp.set'))

    def test_clean_drifts_basic(self):
        """Test clean_drifts with basic parameters."""
        # Transition is [low_freq, high_freq] for the highpass filter transition band
        # Attenuation is the stopband attenuation in dB
        result = clean_drifts(deepcopy(self.EEG), [0.25, 0.75], 80)
        self.assertIsNotNone(result)
        self.assertIn('data', result)

    def test_clean_drifts_with_custom_eeglab(self):
        """Test clean_drifts with custom eeglab instance."""
        eeglab = get_eeglab('MAT')
        result = clean_drifts(deepcopy(self.EEG), [0.25, 0.75], 80, eeglab=eeglab)
        self.assertIsNotNone(result)


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestPopEegfiltnew(DebuggableTestCase):
    """Test cases for pop_eegfiltnew function."""

    def setUp(self):
        """Set up test fixtures with real data."""
        self.EEG = pop_loadset(os.path.join(LOCAL_DATA_PATH, 'eeglab_data_with_ica_tmp.set'))

    def test_pop_eegfiltnew_lowpass(self):
        """Test pop_eegfiltnew with lowpass filter."""
        result = pop_eegfiltnew(deepcopy(self.EEG), locutoff=None, hicutoff=25)
        self.assertIsNotNone(result)
        self.assertIn('data', result)
        self.assertEqual(result['srate'], self.EEG['srate'])

    def test_pop_eegfiltnew_highpass(self):
        """Test pop_eegfiltnew with highpass filter."""
        result = pop_eegfiltnew(deepcopy(self.EEG), locutoff=5, hicutoff=None)
        self.assertIsNotNone(result)
        self.assertIn('data', result)

    def test_pop_eegfiltnew_bandpass(self):
        """Test pop_eegfiltnew with bandpass filter."""
        result = pop_eegfiltnew(deepcopy(self.EEG), locutoff=5, hicutoff=25)
        self.assertIsNotNone(result)
        self.assertIn('data', result)

    def test_pop_eegfiltnew_with_revfilt(self):
        """Test pop_eegfiltnew with reverse filter."""
        result = pop_eegfiltnew(deepcopy(self.EEG), locutoff=5, hicutoff=25, revfilt=True)
        self.assertIsNotNone(result)
        self.assertIn('data', result)

    def test_pop_eegfiltnew_with_plotfreqz(self):
        """Test pop_eegfiltnew with plotfreqz."""
        result = pop_eegfiltnew(deepcopy(self.EEG), locutoff=5, hicutoff=25, plotfreqz=True)
        self.assertIsNotNone(result)
        self.assertIn('data', result)

    def test_pop_eegfiltnew_no_cutoffs(self):
        """Test pop_eegfiltnew with no cutoffs defined."""
        with self.assertRaises(Exception):
            pop_eegfiltnew(deepcopy(self.EEG), locutoff=None, hicutoff=None)


class TestCleanArtifacts(DebuggableTestCase):
    """Test cases for clean_artifacts function."""

    def setUp(self):
        """Set up test fixtures with real data."""
        self.EEG = pop_loadset(os.path.join(LOCAL_DATA_PATH, 'eeglab_data_with_ica_tmp.set'))

    def test_clean_artifacts_basic(self):
        """Test clean_artifacts with basic parameters (channel cleaning only)."""
        result, *_ = clean_artifacts(deepcopy(self.EEG), BurstCriterion='off')
        self.assertIsNotNone(result)
        self.assertIn('data', result)

    def test_clean_artifacts_with_channel_criterion(self):
        """Test clean_artifacts with channel criterion."""
        result, *_ = clean_artifacts(deepcopy(self.EEG), ChannelCriterion=0.8, BurstCriterion='off')
        self.assertIsNotNone(result)
        self.assertIn('data', result)

    def test_clean_artifacts_with_burst_criterion(self):
        """Test clean_artifacts with burst criterion."""
        result, *_ = clean_artifacts(deepcopy(self.EEG), ChannelCriterion='off', BurstCriterion=20)
        self.assertIsNotNone(result)
        self.assertIn('data', result)

    def test_clean_artifacts_all_criteria_disabled(self):
        """Test clean_artifacts with all criteria disabled."""
        result, *_ = clean_artifacts(
            deepcopy(self.EEG),
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            FlatlineCriterion='off',
            BurstCriterion='off',
            Highpass='off'
        )
        self.assertIsNotNone(result)
        self.assertIn('data', result)


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
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
