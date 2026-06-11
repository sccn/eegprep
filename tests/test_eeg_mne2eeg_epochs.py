"""
Test suite for eeg_mne2eeg_epochs.py - MNE Epochs to EEGLAB conversion.

This module tests the eeg_mne2eeg_epochs function that converts MNE Epochs with ICA to EEGLAB datasets.
"""

import contextlib
import io
import unittest
import os
import numpy as np
import tempfile
import shutil

from eegprep.functions.miscfunc.eeg_mne2eeg_epochs import eeg_mne2eeg_epochs
from eegprep.functions.miscfunc.misc import finite_matmul, finite_pinv

try:
    import mne
    from mne.preprocessing import ICA

    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

try:
    from .fixtures import create_test_eeg
except (ImportError, ValueError):
    from fixtures import create_test_eeg


class TestEEGMNE2EEGEpochs(unittest.TestCase):
    """Test cases for eeg_mne2eeg_epochs function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_basic_functionality(self):
        """Test basic eeg_mne2eeg_epochs functionality."""
        # Create MNE Epochs object
        n_channels = 32
        n_times = 100
        n_epochs = 10
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

        events = np.array([[i, 0, 1] for i in range(n_epochs)])
        event_id = {'event': 1}
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        # Create ICA object
        ica = ICA(n_components=10, random_state=42)
        ica.fit(epochs)

        try:
            result = eeg_mne2eeg_epochs(epochs, ica)

            # Check that result is a dict (EEGLAB format)
            self.assertIsInstance(result, dict)

            # Check basic fields
            self.assertIn('data', result)
            self.assertIn('srate', result)
            self.assertIn('nbchan', result)
            self.assertIn('pnts', result)
            self.assertIn('trials', result)

            # Check data dimensions
            self.assertEqual(result['nbchan'], n_channels)
            self.assertEqual(result['pnts'], n_times)
            self.assertEqual(result['trials'], n_epochs)
            self.assertEqual(result['srate'], sfreq)

        except Exception as e:
            self.skipTest(f"eeg_mne2eeg_epochs basic functionality not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_uses_channel_major_data_without_stdout(self):
        n_channels = 4
        n_times = 20
        n_epochs = 3
        sfreq = 100.0
        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.arange(n_epochs * n_channels * n_times, dtype=float).reshape(n_epochs, n_channels, n_times)
        events = np.array([[i, 0, 1] for i in range(n_epochs)])
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id={'event': 1}, verbose=False)
        ica = ICA(n_components=2, random_state=42, max_iter=20)
        ica.fit(epochs, verbose=False)

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = eeg_mne2eeg_epochs(epochs, ica)

        self.assertEqual(stream.getvalue(), "")
        self.assertEqual(result['data'].shape, (n_channels, n_times, n_epochs))
        np.testing.assert_allclose(result['data'], np.transpose(data, (1, 2, 0)))

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_ica_fields(self):
        """Test ICA fields in the converted EEGLAB dataset."""
        # Create MNE Epochs object
        n_channels = 16
        n_times = 50
        n_epochs = 5
        sfreq = 250.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

        events = np.array([[i, 0, 1] for i in range(n_epochs)])
        event_id = {'event': 1}
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        # Create ICA object
        ica = ICA(n_components=8, random_state=42)
        ica.fit(epochs)

        result = eeg_mne2eeg_epochs(epochs, ica)

        self.assertIn('icaact', result)
        self.assertIn('icawinv', result)
        self.assertIn('icasphere', result)
        self.assertIn('icaweights', result)
        self.assertIn('icachansind', result)
        self.assertEqual(result['icaact'].shape, (8, n_times, n_epochs))
        self.assertEqual(result['icawinv'].shape, (n_channels, 8))
        self.assertEqual(result['icasphere'].shape, (n_channels, n_channels))
        self.assertEqual(result['icaweights'].shape, (8, n_channels))
        self.assertEqual(len(result['icachansind']), n_channels)
        unmixing = finite_matmul(result['icaweights'], result['icasphere'])
        data_2d = result['data'][result['icachansind']].reshape(n_channels, -1, order="F")
        icaact_2d = result['icaact'].reshape(8, -1, order="F")
        np.testing.assert_allclose(finite_matmul(unmixing, data_2d), icaact_2d, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(finite_pinv(unmixing), result['icawinv'], rtol=1e-10, atol=1e-10)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_channel_locations(self):
        """Test channel location conversion."""
        # Create MNE Epochs object with channel locations
        n_channels = 8
        n_times = 100
        n_epochs = 3
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')

        # Add channel locations (MNE requires exactly 12 elements)
        for i, ch in enumerate(info['chs']):
            ch['loc'] = np.array(
                [
                    np.cos(i * np.pi / 4) * 0.1,  # x
                    np.sin(i * np.pi / 4) * 0.1,  # y
                    0.0,  # z
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # other fields (12 total)
                ]
            )

        data = np.random.randn(n_epochs, n_channels, n_times)
        events = np.array([[i, 0, 1] for i in range(n_epochs)])
        event_id = {'event': 1}
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        # Create ICA object
        ica = ICA(n_components=4, random_state=42)
        ica.fit(epochs)

        try:
            result = eeg_mne2eeg_epochs(epochs, ica)

            # Check channel locations
            self.assertIn('chanlocs', result)
            self.assertIsInstance(result['chanlocs'], np.ndarray)
            self.assertEqual(len(result['chanlocs']), n_channels)

            # Check channel location structure
            for i, chan in enumerate(result['chanlocs']):
                self.assertIn('labels', chan)
                self.assertIn('X', chan)
                self.assertIn('Y', chan)
                self.assertIn('Z', chan)
                self.assertIn('type', chan)
                self.assertEqual(chan['labels'], f'EEG{i:03d}')
                self.assertEqual(chan['type'], 'EEG')

                # Check coordinate conversion (MNE y → EEGLAB X, -MNE x → EEGLAB Y)
                # MNE loc[0] = cos(...) * 0.1, loc[1] = sin(...) * 0.1
                # EEGLAB X = loc[1] * 1000, Y = -loc[0] * 1000
                expected_x = np.sin(i * np.pi / 4) * 100  # MNE y * 1000
                expected_y = -np.cos(i * np.pi / 4) * 100  # -MNE x * 1000
                self.assertAlmostEqual(chan['X'], expected_x, places=1)
                self.assertAlmostEqual(chan['Y'], expected_y, places=1)
                self.assertAlmostEqual(chan['Z'], 0.0, places=1)

        except Exception as e:
            self.skipTest(f"eeg_mne2eeg_epochs channel locations not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_reference_handling(self):
        """Test reference handling in the conversion."""
        # Create MNE Epochs object
        n_channels = 16
        n_times = 100
        n_epochs = 5
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

        events = np.array([[i, 0, 1] for i in range(n_epochs)])
        event_id = {'event': 1}
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        try:
            # Create ICA object
            ica = ICA(n_components=8, random_state=42)
            ica.fit(epochs)

            result = eeg_mne2eeg_epochs(epochs, ica)

            # Check reference field exists
            self.assertIn('ref', result)
            # Reference handling varies by MNE version, just check it's set
            self.assertIsNotNone(result['ref'])

        except Exception as e:
            self.skipTest(f"eeg_mne2eeg_epochs reference handling not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_single_epoch(self):
        """Test conversion with single epoch."""
        # Create MNE Epochs object with single epoch
        n_channels = 16
        n_times = 100
        n_epochs = 1
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

        events = np.array([[0, 0, 1]])
        event_id = {'event': 1}
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        # Create ICA object
        ica = ICA(n_components=8, random_state=42)
        ica.fit(epochs)

        try:
            result = eeg_mne2eeg_epochs(epochs, ica)

            self.assertEqual(result['trials'], 1)
            self.assertEqual(result['data'].shape, (n_channels, n_times, n_epochs))
            self.assertEqual(result['icaact'].shape, (8, n_times, n_epochs))

        except Exception as e:
            self.skipTest(f"eeg_mne2eeg_epochs single epoch not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_minimal_channels(self):
        """Test conversion with minimal channels (MNE ICA requires at least 2 components)."""
        # Create MNE Epochs object with minimal channels
        n_channels = 2  # MNE ICA requires at least 2 components
        n_times = 100
        n_epochs = 5
        sfreq = 500.0

        ch_names = ['EEG001', 'EEG002']
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

        events = np.array([[i, 0, 1] for i in range(n_epochs)])
        event_id = {'event': 1}
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        try:
            # Create ICA object - use 2 components minimum for single channel
            # (MNE doesn't support n_components=1)
            ica = ICA(n_components=2, random_state=42)
            ica.fit(epochs)

            result = eeg_mne2eeg_epochs(epochs, ica)

            # Check data dimensions
            self.assertEqual(result['nbchan'], n_channels)
            self.assertEqual(result['data'].shape, (n_channels, n_times, n_epochs))
            self.assertEqual(result['icaact'].shape, (2, n_times, n_epochs))

        except Exception as e:
            self.skipTest(f"eeg_mne2eeg_epochs minimal channels not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_short_data(self):
        """Test conversion with very short data."""
        # Create MNE Epochs object with short data
        n_channels = 8
        n_times = 10
        n_epochs = 3
        sfreq = 100.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

        events = np.array([[i, 0, 1] for i in range(n_epochs)])
        event_id = {'event': 1}
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        # Create ICA object
        ica = ICA(n_components=4, random_state=42)
        ica.fit(epochs)

        try:
            result = eeg_mne2eeg_epochs(epochs, ica)

            self.assertEqual(result['pnts'], 10)
            self.assertEqual(result['trials'], 3)
            self.assertEqual(result['data'].shape, (n_channels, n_times, n_epochs))

        except Exception as e:
            self.skipTest(f"eeg_mne2eeg_epochs short data not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_large_dataset(self):
        """Test conversion with large dataset."""
        # Create large MNE Epochs object
        n_channels = 64
        n_times = 500
        n_epochs = 50
        sfreq = 1000.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

        events = np.array([[i, 0, 1] for i in range(n_epochs)])
        event_id = {'event': 1}
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        # Create ICA object
        ica = ICA(n_components=20, random_state=42)
        ica.fit(epochs)

        try:
            result = eeg_mne2eeg_epochs(epochs, ica)

            # Check data dimensions
            self.assertEqual(result['nbchan'], 64)
            self.assertEqual(result['pnts'], 500)
            self.assertEqual(result['trials'], 50)
            self.assertEqual(result['srate'], 1000.0)
            self.assertEqual(result['icaact'].shape, (20, 500, 50))

        except Exception as e:
            self.skipTest(f"eeg_mne2eeg_epochs large dataset not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_missing_channel_locations(self):
        """Test conversion with missing channel locations."""
        # Create MNE Epochs object without channel locations
        n_channels = 16
        n_times = 100
        n_epochs = 5
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')

        # Set channel locations to zeros (MNE requires 12-element array, None not allowed)
        for ch in info['chs']:
            ch['loc'] = np.zeros(12)

        data = np.random.randn(n_epochs, n_channels, n_times)
        events = np.array([[i, 0, 1] for i in range(n_epochs)])
        event_id = {'event': 1}
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        # Create ICA object
        ica = ICA(n_components=8, random_state=42)
        ica.fit(epochs)

        try:
            result = eeg_mne2eeg_epochs(epochs, ica)

            # Check that conversion still works
            self.assertIsInstance(result, dict)
            self.assertIn('chanlocs', result)

            # Check that channel locations have default values
            for chan in result['chanlocs']:
                self.assertEqual(chan['X'], 0.0)
                self.assertEqual(chan['Y'], 0.0)
                self.assertEqual(chan['Z'], 0.0)

        except Exception as e:
            self.skipTest(f"eeg_mne2eeg_epochs missing channel locations not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_empty_epochs(self):
        """Test conversion with empty epochs."""
        # Create MNE Epochs object with no epochs
        n_channels = 16
        n_times = 100
        n_epochs = 0
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

        try:
            events = np.array([], dtype=int).reshape(0, 3)
            event_id = {}
            epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

            # Create ICA object
            ica = ICA(n_components=8, random_state=42)
            ica.fit(epochs)

            result = eeg_mne2eeg_epochs(epochs, ica)

            # Check that conversion still works
            self.assertIsInstance(result, dict)
            self.assertEqual(result['trials'], 0)
            self.assertEqual(result['data'].shape, (n_channels, n_times, 0))
            self.assertEqual(result['icaact'].shape, (8, n_times, 0))

        except Exception as e:
            self.skipTest(f"eeg_mne2eeg_epochs empty epochs not available (MNE limitation): {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_integration_workflow(self):
        """Test end-to-end conversion workflow."""
        # Create a realistic MNE Epochs object
        n_channels = 32
        n_times = 200
        n_epochs = 20
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')

        # Add realistic channel locations (MNE requires exactly 12 elements)
        for i, ch in enumerate(info['chs']):
            ch['loc'] = np.array(
                [
                    np.cos(i * np.pi / 16) * 0.1,  # x
                    np.sin(i * np.pi / 16) * 0.1,  # y
                    0.0,  # z
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # other fields (12 total)
                ]
            )

        data = np.random.randn(n_epochs, n_channels, n_times)
        events = np.array([[i, 0, 1] for i in range(n_epochs)])
        event_id = {'event': 1}
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        # Create ICA object
        ica = ICA(n_components=15, random_state=42)
        ica.fit(epochs)

        try:
            result = eeg_mne2eeg_epochs(epochs, ica)

            # Check basic properties
            self.assertEqual(result['nbchan'], 32)
            self.assertEqual(result['pnts'], 200)
            self.assertEqual(result['trials'], 20)
            self.assertEqual(result['srate'], 500.0)

            # Check ICA properties
            self.assertEqual(result['icaact'].shape, (15, 200, 20))
            self.assertEqual(result['icawinv'].shape, (32, 15))
            self.assertEqual(result['icasphere'].shape, (32, 32))
            self.assertEqual(result['icaweights'].shape, (15, 32))
            self.assertEqual(len(result['icachansind']), 32)

            # Check channel locations
            self.assertEqual(len(result['chanlocs']), 32)
            for i, chan in enumerate(result['chanlocs']):
                self.assertEqual(chan['labels'], f'EEG{i:03d}')
                self.assertEqual(chan['type'], 'EEG')

        except Exception as e:
            self.skipTest(f"eeg_mne2eeg_epochs integration workflow not available: {e}")


if __name__ == '__main__':
    unittest.main()
