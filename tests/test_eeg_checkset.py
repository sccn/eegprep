"""
Test suite for eeg_checkset.py - EEG structure validation and normalization.

This module tests the eeg_checkset function that validates and normalizes
EEG data structures, ensuring required fields exist and have correct types.
"""

import unittest
import sys
import numpy as np
import tempfile
import os

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.eeg_checkset import eeg_checkset, strict_mode
from eegprep.utils.testing import DebuggableTestCase


def create_minimal_eeg():
    """Create a minimal EEG structure with only required fields."""
    return {
        'data': np.random.randn(32, 1000),  # 2D continuous data
        'srate': 250.0,
        'xmin': 0.0,
        'xmax': 4.0,
    }


def create_complete_eeg():
    """Create a complete EEG structure with all standard fields."""
    n_channels = 32
    n_samples = 1000
    n_trials = 1

    return {
        'data': np.random.randn(n_channels, n_samples),
        'srate': 250.0,
        'nbchan': n_channels,
        'pnts': n_samples,
        'trials': n_trials,
        'xmin': 0.0,
        'xmax': 4.0,
        'times': np.linspace(0, 4.0, n_samples),
        'setname': 'test_dataset',
        'filename': 'test.set',
        'filepath': '/tmp',
        'subject': '',
        'group': '',
        'condition': '',
        'session': 1,
        'comments': np.array([]),
        'chanlocs': np.array([{'labels': f'Ch{i+1}'} for i in range(n_channels)]),
        'urchanlocs': np.array([]),
        'chaninfo': {},
        'ref': 'common',
        'event': np.array([]),
        'urevent': np.array([]),
        'eventdescription': np.array([]),
        'epoch': np.array([]),
        'epochdescription': np.array([]),
        'reject': {},
        'stats': {},
        'specdata': {},
        'specicaact': {},
        'splinefile': '',
        'icasplinefile': '',
        'dipfit': {},
        'history': '',
        'saved': 'yes',
        'etc': {},
        'datfile': '',
        'run': 1,
        'roi': {},
        'icaact': np.array([]),
        'icawinv': np.array([]),
        'icasphere': np.array([]),
        'icaweights': np.array([]),
        'icachansind': np.array([])
    }


class TestEegChecksetBasic(DebuggableTestCase):
    """Basic test cases for eeg_checkset function."""

    def test_eeg_checkset_minimal_structure(self):
        """Test eeg_checkset with minimal EEG structure."""
        eeg = create_minimal_eeg()
        result = eeg_checkset(eeg)

        # Check that required fields are present
        self.assertIn('data', result)
        self.assertIn('srate', result)
        self.assertIn('nbchan', result)
        self.assertIn('pnts', result)
        self.assertIn('trials', result)

        # Check that fields were inferred correctly
        self.assertEqual(result['nbchan'], 32)
        self.assertEqual(result['pnts'], 1000)
        self.assertEqual(result['trials'], 1)

    def test_eeg_checkset_complete_structure(self):
        """Test eeg_checkset with complete EEG structure."""
        eeg = create_complete_eeg()
        result = eeg_checkset(eeg)

        # Check that all fields are preserved
        self.assertIn('data', result)
        self.assertIn('srate', result)
        self.assertIn('nbchan', result)
        self.assertIn('chanlocs', result)

        # Check values are correct
        self.assertEqual(result['nbchan'], 32)
        self.assertEqual(result['pnts'], 1000)
        self.assertEqual(result['trials'], 1)

    def test_eeg_checkset_preserves_data(self):
        """Test that eeg_checkset preserves original data."""
        eeg = create_minimal_eeg()
        original_data = eeg['data'].copy()

        result = eeg_checkset(eeg)

        # Data should be preserved
        np.testing.assert_array_equal(result['data'], original_data)


class TestEegChecksetFieldInference(DebuggableTestCase):
    """Test field inference and default value assignment."""

    def test_nbchan_inference_from_data(self):
        """Test that nbchan is inferred from data shape when missing."""
        eeg = create_minimal_eeg()
        del eeg['srate']  # Remove to avoid issues
        eeg['srate'] = 250.0
        # Don't set nbchan

        result = eeg_checkset(eeg)

        # nbchan should be inferred from data shape
        self.assertEqual(result['nbchan'], eeg['data'].shape[0])

    def test_pnts_inference_from_data(self):
        """Test that pnts is inferred from data shape when missing."""
        eeg = create_minimal_eeg()
        # Don't set pnts

        result = eeg_checkset(eeg)

        # pnts should be inferred from data shape
        self.assertEqual(result['pnts'], eeg['data'].shape[1])

    def test_trials_inference_2d_data(self):
        """Test that trials is inferred as 1 for 2D data."""
        eeg = create_minimal_eeg()
        eeg['data'] = np.random.randn(32, 1000)  # 2D data

        result = eeg_checkset(eeg)

        # trials should be 1 for 2D data
        self.assertEqual(result['trials'], 1)

    def test_trials_inference_3d_data(self):
        """Test that trials is inferred from data shape for 3D data."""
        eeg = create_minimal_eeg()
        n_trials = 10
        eeg['data'] = np.random.randn(32, 1000, n_trials)  # 3D data

        result = eeg_checkset(eeg)

        # trials should be inferred from 3D data shape
        self.assertEqual(result['trials'], n_trials)

    def test_missing_fields_added_with_defaults(self):
        """Test that missing fields are added with appropriate defaults."""
        eeg = create_minimal_eeg()

        result = eeg_checkset(eeg)

        # Check that missing fields are added
        self.assertIn('setname', result)
        self.assertIn('filename', result)
        self.assertIn('filepath', result)
        self.assertIn('event', result)
        self.assertIn('chanlocs', result)
        self.assertIn('chaninfo', result)

        # Check default values
        self.assertEqual(result['setname'], '')
        self.assertEqual(result['filename'], '')
        self.assertIsInstance(result['event'], np.ndarray)
        self.assertIsInstance(result['chaninfo'], dict)


class TestEegChecksetTypeConversion(DebuggableTestCase):
    """Test type conversion and validation."""

    def test_nbchan_converted_to_int(self):
        """Test that nbchan is converted to int."""
        eeg = create_minimal_eeg()
        eeg['nbchan'] = 32.0  # Float

        result = eeg_checkset(eeg)

        self.assertEqual(result['nbchan'], 32)
        self.assertIsInstance(result['nbchan'], int)

    def test_pnts_converted_to_int(self):
        """Test that pnts is converted to int."""
        eeg = create_minimal_eeg()
        eeg['pnts'] = 1000.0  # Float

        result = eeg_checkset(eeg)

        self.assertEqual(result['pnts'], 1000)
        self.assertIsInstance(result['pnts'], int)

    def test_trials_converted_to_int(self):
        """Test that trials is converted to int."""
        eeg = create_minimal_eeg()
        eeg['trials'] = 1.0  # Float

        result = eeg_checkset(eeg)

        self.assertEqual(result['trials'], 1)
        self.assertIsInstance(result['trials'], int)

    def test_xmin_converted_to_float(self):
        """Test that xmin is converted to float."""
        eeg = create_minimal_eeg()
        eeg['xmin'] = 0  # Int

        result = eeg_checkset(eeg)

        self.assertEqual(result['xmin'], 0.0)
        self.assertIsInstance(result['xmin'], float)

    def test_xmax_converted_to_float(self):
        """Test that xmax is converted to float."""
        eeg = create_minimal_eeg()
        eeg['xmax'] = 4  # Int

        result = eeg_checkset(eeg)

        self.assertEqual(result['xmax'], 4.0)
        self.assertIsInstance(result['xmax'], float)

    def test_srate_converted_to_float(self):
        """Test that srate is converted to float."""
        eeg = create_minimal_eeg()
        eeg['srate'] = 250  # Int

        result = eeg_checkset(eeg)

        self.assertEqual(result['srate'], 250.0)
        self.assertIsInstance(result['srate'], float)


class TestEegChecksetEventHandling(DebuggableTestCase):
    """Test event handling and conversion."""

    def test_event_dict_to_array(self):
        """Test that single event dict is converted to array."""
        eeg = create_minimal_eeg()
        eeg['event'] = {'type': 'stimulus', 'latency': 100}

        result = eeg_checkset(eeg)

        # Should be converted to numpy array
        self.assertIsInstance(result['event'], np.ndarray)
        self.assertEqual(len(result['event']), 1)

    def test_event_list_to_array(self):
        """Test that event list is converted to numpy array."""
        eeg = create_minimal_eeg()
        eeg['event'] = [
            {'type': 'stimulus', 'latency': 100},
            {'type': 'response', 'latency': 200}
        ]

        result = eeg_checkset(eeg)

        # Should be converted to numpy array
        self.assertIsInstance(result['event'], np.ndarray)
        self.assertEqual(len(result['event']), 2)

    def test_event_missing_creates_empty_array(self):
        """Test that missing event field creates empty array."""
        eeg = create_minimal_eeg()
        # Don't set event

        result = eeg_checkset(eeg)

        # Should create empty array
        self.assertIsInstance(result['event'], np.ndarray)
        self.assertEqual(len(result['event']), 0)


class TestEegChecksetChanlocsHandling(DebuggableTestCase):
    """Test channel location handling and conversion."""

    def test_chanlocs_dict_to_array(self):
        """Test that single chanlocs dict is converted to array."""
        eeg = create_minimal_eeg()
        eeg['chanlocs'] = {'labels': 'Ch1', 'theta': 0, 'radius': 0.5}

        result = eeg_checkset(eeg)

        # Should be converted to numpy array
        self.assertIsInstance(result['chanlocs'], np.ndarray)
        self.assertEqual(len(result['chanlocs']), 1)

    def test_chanlocs_list_to_array(self):
        """Test that chanlocs list is converted to numpy array."""
        eeg = create_minimal_eeg()
        eeg['chanlocs'] = [
            {'labels': 'Ch1', 'theta': 0},
            {'labels': 'Ch2', 'theta': 45}
        ]

        result = eeg_checkset(eeg)

        # Should be converted to numpy array
        self.assertIsInstance(result['chanlocs'], np.ndarray)
        self.assertEqual(len(result['chanlocs']), 2)

    def test_chanlocs_missing_creates_empty_array(self):
        """Test that missing chanlocs field creates empty array."""
        eeg = create_minimal_eeg()
        # Don't set chanlocs

        result = eeg_checkset(eeg)

        # Should create empty array
        self.assertIsInstance(result['chanlocs'], np.ndarray)
        self.assertEqual(len(result['chanlocs']), 0)


class TestEegChecksetChaninfoHandling(DebuggableTestCase):
    """Test chaninfo handling."""

    def test_chaninfo_missing_creates_empty_dict(self):
        """Test that missing chaninfo field creates empty dict."""
        eeg = create_minimal_eeg()
        # Don't set chaninfo

        result = eeg_checkset(eeg)

        # Should create empty dict
        self.assertIsInstance(result['chaninfo'], dict)
        self.assertEqual(len(result['chaninfo']), 0)

    def test_chaninfo_preserved(self):
        """Test that existing chaninfo is preserved."""
        eeg = create_minimal_eeg()
        eeg['chaninfo'] = {'removedchans': [1, 2, 3]}

        result = eeg_checkset(eeg)

        # Should be preserved
        self.assertIn('removedchans', result['chaninfo'])
        self.assertEqual(result['chaninfo']['removedchans'], [1, 2, 3])


class TestEegChecksetRejectHandling(DebuggableTestCase):
    """Test reject handling."""

    def test_reject_missing_creates_empty_dict(self):
        """Test that missing reject field creates empty dict."""
        eeg = create_minimal_eeg()
        # Don't set reject

        result = eeg_checkset(eeg)

        # Should create empty dict
        self.assertIsInstance(result['reject'], dict)
        self.assertEqual(len(result['reject']), 0)

    def test_reject_preserved(self):
        """Test that existing reject is preserved."""
        eeg = create_minimal_eeg()
        eeg['reject'] = {'rejthresh': 100}

        result = eeg_checkset(eeg)

        # Should be preserved
        self.assertIn('rejthresh', result['reject'])
        self.assertEqual(result['reject']['rejthresh'], 100)


class TestEegChecksetDataSqueezing(DebuggableTestCase):
    """Test 3D data squeezing behavior."""

    def test_3d_data_single_trial_squeezed(self):
        """Test that 3D data with single trial is squeezed to 2D."""
        eeg = create_minimal_eeg()
        eeg['data'] = np.random.randn(32, 1000, 1)  # 3D with single trial

        result = eeg_checkset(eeg)

        # Should be squeezed to 2D
        self.assertEqual(result['data'].ndim, 2)
        self.assertEqual(result['data'].shape, (32, 1000))

    def test_3d_data_multiple_trials_not_squeezed(self):
        """Test that 3D data with multiple trials is not squeezed."""
        eeg = create_minimal_eeg()
        eeg['data'] = np.random.randn(32, 1000, 5)  # 3D with 5 trials

        result = eeg_checkset(eeg)

        # Should remain 3D
        self.assertEqual(result['data'].ndim, 3)
        self.assertEqual(result['data'].shape, (32, 1000, 5))

    def test_2d_data_not_modified(self):
        """Test that 2D data is not modified."""
        eeg = create_minimal_eeg()
        eeg['data'] = np.random.randn(32, 1000)  # 2D

        result = eeg_checkset(eeg)

        # Should remain 2D
        self.assertEqual(result['data'].ndim, 2)
        self.assertEqual(result['data'].shape, (32, 1000))


class TestEegChecksetICAActivations(DebuggableTestCase):
    """Test ICA activation computation."""

    def test_ica_activations_computed(self):
        """Test that ICA activations are computed when weights and sphere exist."""
        eeg = create_minimal_eeg()
        n_components = 8

        # Add ICA weights and sphere
        eeg['icaweights'] = np.random.randn(n_components, 32)
        eeg['icasphere'] = np.eye(32)

        result = eeg_checkset(eeg)

        # ICA activations should be computed
        self.assertIn('icaact', result)
        self.assertIsInstance(result['icaact'], np.ndarray)
        self.assertGreater(result['icaact'].size, 0)
        self.assertEqual(result['icaact'].shape[0], n_components)

    def test_ica_activations_not_computed_missing_weights(self):
        """Test that ICA activations are not computed when weights are missing."""
        eeg = create_minimal_eeg()
        eeg['icasphere'] = np.eye(32)
        # Don't set icaweights

        result = eeg_checkset(eeg)

        # ICA activations should be empty
        self.assertIn('icaact', result)
        self.assertEqual(result['icaact'].size, 0)

    def test_ica_activations_not_computed_missing_sphere(self):
        """Test that ICA activations are not computed when sphere is missing."""
        eeg = create_minimal_eeg()
        eeg['icaweights'] = np.random.randn(8, 32)
        # Don't set icasphere

        result = eeg_checkset(eeg)

        # ICA activations should be empty
        self.assertIn('icaact', result)
        self.assertEqual(result['icaact'].size, 0)

    def test_ica_activations_not_computed_empty_weights(self):
        """Test that ICA activations are not computed when weights are empty."""
        eeg = create_minimal_eeg()
        eeg['icaweights'] = np.array([])
        eeg['icasphere'] = np.eye(32)

        result = eeg_checkset(eeg)

        # ICA activations should be empty
        self.assertIn('icaact', result)
        self.assertEqual(result['icaact'].size, 0)

    def test_ica_activations_float32_dtype(self):
        """Test that ICA activations are converted to float32."""
        eeg = create_minimal_eeg()
        n_components = 8

        eeg['icaweights'] = np.random.randn(n_components, 32)
        eeg['icasphere'] = np.eye(32)

        result = eeg_checkset(eeg)

        # Should be float32
        self.assertEqual(result['icaact'].dtype, np.float32)


class TestEegChecksetStrictMode(DebuggableTestCase):
    """Test strict mode functionality."""

    def test_strict_mode_enabled_by_default(self):
        """Test that strict mode is enabled by default."""
        eeg = create_minimal_eeg()
        # Create invalid ICA setup that would cause error
        eeg['icaweights'] = np.random.randn(8, 32)
        eeg['icasphere'] = np.random.randn(16, 16)  # Wrong size

        # Should raise error in strict mode
        with self.assertRaises(Exception):
            eeg_checkset(eeg)

    def test_strict_mode_disabled_catches_errors(self):
        """Test that disabled strict mode catches and handles errors gracefully."""
        eeg = create_minimal_eeg()
        # Create invalid ICA setup that would cause error
        eeg['icaweights'] = np.random.randn(8, 32)
        eeg['icasphere'] = np.random.randn(16, 16)  # Wrong size

        # Should not raise error when strict mode is disabled
        with strict_mode(False):
            result = eeg_checkset(eeg)
            # Should return result with empty icaact
            self.assertIn('icaact', result)
            self.assertEqual(result['icaact'].size, 0)

    def test_strict_mode_context_manager(self):
        """Test that strict mode context manager works correctly."""
        eeg1 = create_minimal_eeg()
        eeg2 = create_minimal_eeg()

        # Add invalid ICA setup
        eeg1['icaweights'] = np.random.randn(8, 32)
        eeg1['icasphere'] = np.random.randn(16, 16)  # Wrong size

        # Should work inside strict_mode(False)
        with strict_mode(False):
            result1 = eeg_checkset(eeg1)
            self.assertIsNotNone(result1)

        # Should revert to strict mode after context
        eeg2['icaweights'] = np.random.randn(8, 32)
        eeg2['icasphere'] = np.random.randn(16, 16)  # Wrong size

        with self.assertRaises(Exception):
            eeg_checkset(eeg2)


class TestEegChecksetEdgeCases(DebuggableTestCase):
    """Test edge cases and unusual inputs."""

    def test_single_channel(self):
        """Test eeg_checkset with single channel data."""
        eeg = create_minimal_eeg()
        eeg['data'] = np.random.randn(1, 1000)

        result = eeg_checkset(eeg)

        self.assertEqual(result['nbchan'], 1)
        self.assertEqual(result['pnts'], 1000)

    def test_single_sample(self):
        """Test eeg_checkset with single sample data."""
        eeg = create_minimal_eeg()
        eeg['data'] = np.random.randn(32, 1)

        result = eeg_checkset(eeg)

        self.assertEqual(result['nbchan'], 32)
        self.assertEqual(result['pnts'], 1)

    def test_very_short_data(self):
        """Test eeg_checkset with very short data."""
        eeg = create_minimal_eeg()
        eeg['data'] = np.random.randn(32, 10)

        result = eeg_checkset(eeg)

        self.assertEqual(result['nbchan'], 32)
        self.assertEqual(result['pnts'], 10)

    def test_many_trials(self):
        """Test eeg_checkset with many trials."""
        eeg = create_minimal_eeg()
        eeg['data'] = np.random.randn(32, 100, 50)  # 50 trials

        result = eeg_checkset(eeg)

        self.assertEqual(result['trials'], 50)


class TestEegChecksetIntegration(DebuggableTestCase):
    """Integration tests for eeg_checkset."""

    def test_complete_workflow(self):
        """Test complete workflow with realistic EEG structure."""
        eeg = {
            'data': np.random.randn(64, 2000),
            'srate': 500.0,
            'xmin': -0.5,
            'xmax': 3.5,
            'setname': 'Subject01_Session01',
            'event': [
                {'type': 'stimulus', 'latency': 250},
                {'type': 'response', 'latency': 500}
            ],
            'chanlocs': [{'labels': f'EEG{i:03d}'} for i in range(1, 65)]
        }

        result = eeg_checkset(eeg)

        # Verify all fields are properly set
        self.assertEqual(result['nbchan'], 64)
        self.assertEqual(result['pnts'], 2000)
        self.assertEqual(result['trials'], 1)
        self.assertEqual(result['srate'], 500.0)
        self.assertIsInstance(result['event'], np.ndarray)
        self.assertEqual(len(result['event']), 2)
        self.assertIsInstance(result['chanlocs'], np.ndarray)
        self.assertEqual(len(result['chanlocs']), 64)

    def test_minimal_to_complete_conversion(self):
        """Test that minimal structure is expanded to complete structure."""
        eeg = create_minimal_eeg()

        result = eeg_checkset(eeg)

        # All standard fields should be present
        required_fields = [
            'data', 'srate', 'nbchan', 'pnts', 'trials',
            'xmin', 'xmax', 'setname', 'filename', 'filepath',
            'event', 'chanlocs', 'chaninfo', 'reject'
        ]

        for field in required_fields:
            self.assertIn(field, result)


if __name__ == '__main__':
    unittest.main()
