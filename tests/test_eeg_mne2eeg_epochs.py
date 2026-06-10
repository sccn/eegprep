"""
Test suite for eeg_mne2eeg_epochs.py - MNE Epochs to EEGLAB conversion.

This module tests the eeg_mne2eeg_epochs function that converts MNE Epochs with ICA to EEGLAB datasets.
"""

import contextlib
import io
import os
import shutil
import tempfile
import unittest

import numpy as np
import pytest

from eegprep.functions.miscfunc.eeg_mne2eeg_epochs import eeg_mne2eeg_epochs

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

# Tests in this file exercise pure-Python MNE conversion; they do not require MATLAB.

XFAIL_178 = pytest.mark.xfail(
    strict=True,
    reason="Converter stores MNE-order data and broken ICA fields; tracked in #178.",
)
XFAIL_178_STDOUT = pytest.mark.xfail(
    strict=True,
    reason="Converter prints a reference-conversion warning to stdout; tracked in #178.",
)


def _make_epochs(n_channels=8, n_times=64, n_epochs=4, sfreq=200.0):
    """Build a small MNE Epochs object with deterministic data."""
    rng = np.random.default_rng(0)
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    data = rng.standard_normal((n_epochs, n_channels, n_times))
    events = np.array([[i, 0, 1] for i in range(n_epochs)])
    return mne.EpochsArray(data, info, events, tmin=0, event_id={'event': 1})


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

        result = eeg_mne2eeg_epochs(epochs, ica)

        self.assertIsInstance(result, dict)
        for key in ('data', 'srate', 'nbchan', 'pnts', 'trials'):
            self.assertIn(key, result)

        self.assertEqual(result['nbchan'], n_channels)
        self.assertEqual(result['pnts'], n_times)
        self.assertEqual(result['trials'], n_epochs)
        self.assertEqual(result['srate'], sfreq)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    @XFAIL_178
    def test_eeg_mne2eeg_epochs_ica_fields(self):
        """ICA fields use EEGLAB shapes: icawinv (chans, comps), icaweights (comps, chans),
        icasphere (chans, chans), icaact (comps, pnts, trials)."""
        n_channels, n_times, n_epochs, n_components = 16, 50, 5, 8
        epochs = _make_epochs(n_channels, n_times, n_epochs, sfreq=250.0)

        ica = ICA(n_components=n_components, random_state=42)
        ica.fit(epochs)

        result = eeg_mne2eeg_epochs(epochs, ica)

        for key in ('icaact', 'icawinv', 'icasphere', 'icaweights', 'icachansind'):
            self.assertIn(key, result)

        # EEGLAB ICA shape contract.
        self.assertEqual(result['icaact'].shape, (n_components, n_times, n_epochs))
        self.assertEqual(result['icawinv'].shape, (n_channels, n_components))
        self.assertEqual(result['icaweights'].shape, (n_components, n_channels))
        self.assertEqual(result['icasphere'].shape, (n_channels, n_channels))
        self.assertEqual(len(result['icachansind']), n_channels)

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

        result = eeg_mne2eeg_epochs(epochs, ica)

        self.assertIn('chanlocs', result)
        self.assertIsInstance(result['chanlocs'], np.ndarray)
        self.assertEqual(len(result['chanlocs']), n_channels)

        for i, chan in enumerate(result['chanlocs']):
            for key in ('labels', 'X', 'Y', 'Z', 'type'):
                self.assertIn(key, chan)
            self.assertEqual(chan['labels'], f'EEG{i:03d}')
            self.assertEqual(chan['type'], 'EEG')

            # MNE loc[0] = cos(...)*0.1, loc[1] = sin(...)*0.1; EEGLAB X = loc[1]*1000, Y = -loc[0]*1000.
            expected_x = np.sin(i * np.pi / 4) * 100
            expected_y = -np.cos(i * np.pi / 4) * 100
            self.assertAlmostEqual(chan['X'], expected_x, places=1)
            self.assertAlmostEqual(chan['Y'], expected_y, places=1)
            self.assertAlmostEqual(chan['Z'], 0.0, places=1)

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

        ica = ICA(n_components=8, random_state=42)
        ica.fit(epochs)

        result = eeg_mne2eeg_epochs(epochs, ica)

        self.assertIn('ref', result)
        # Reference handling varies by MNE version, just check it is populated.
        self.assertIsNotNone(result['ref'])

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    @XFAIL_178
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

        result = eeg_mne2eeg_epochs(epochs, ica)

        # EEGPrep stores epoched data channel-major: (nbchan, pnts, trials).
        self.assertEqual(result['trials'], 1)
        self.assertEqual(result['data'].shape, (n_channels, n_times, n_epochs))
        self.assertEqual(result['icaact'].shape, (8, n_times, n_epochs))

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    @XFAIL_178
    def test_eeg_mne2eeg_epochs_minimal_channels(self):
        """Two-channel ICA exposes the channel-major shape contract."""
        n_channels, n_times, n_epochs = 2, 100, 5
        epochs = _make_epochs(n_channels, n_times, n_epochs, sfreq=500.0)

        ica = ICA(n_components=2, random_state=42)
        ica.fit(epochs)

        result = eeg_mne2eeg_epochs(epochs, ica)

        self.assertEqual(result['nbchan'], n_channels)
        self.assertEqual(result['data'].shape, (n_channels, n_times, n_epochs))
        self.assertEqual(result['icaact'].shape, (2, n_times, n_epochs))

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    @XFAIL_178
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

        result = eeg_mne2eeg_epochs(epochs, ica)

        # EEGPrep stores epoched data channel-major: (nbchan, pnts, trials).
        self.assertEqual(result['pnts'], 10)
        self.assertEqual(result['trials'], 3)
        self.assertEqual(result['data'].shape, (n_channels, n_times, n_epochs))

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

        result = eeg_mne2eeg_epochs(epochs, ica)

        self.assertEqual(result['nbchan'], 64)
        self.assertEqual(result['pnts'], 500)
        self.assertEqual(result['trials'], 50)
        self.assertEqual(result['srate'], 1000.0)
        self.assertEqual(result['icaact'].shape, (20, 500, 50))

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

        result = eeg_mne2eeg_epochs(epochs, ica)

        self.assertIsInstance(result, dict)
        self.assertIn('chanlocs', result)
        for chan in result['chanlocs']:
            self.assertEqual(chan['X'], 0.0)
            self.assertEqual(chan['Y'], 0.0)
            self.assertEqual(chan['Z'], 0.0)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_empty_epochs(self):
        """Empty epochs: MNE refuses to build / fit on 0 epochs."""
        n_channels = 16
        info = mne.create_info([f'EEG{i:03d}' for i in range(n_channels)], 500.0, ch_types='eeg')
        events = np.array([], dtype=int).reshape(0, 3)
        with self.assertRaises((ValueError, RuntimeError)):
            epochs = mne.EpochsArray(np.zeros((0, n_channels, 100)), info, events, tmin=0, event_id={})
            ICA(n_components=8, random_state=42).fit(epochs)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    @XFAIL_178
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

        result = eeg_mne2eeg_epochs(epochs, ica)

        self.assertEqual(result['nbchan'], 32)
        self.assertEqual(result['pnts'], 200)
        self.assertEqual(result['trials'], 20)
        self.assertEqual(result['srate'], 500.0)

        # EEGLAB ICA shape contract: winv (chans, comps), weights (comps, chans), sphere (chans, chans).
        self.assertEqual(result['icaact'].shape, (15, 200, 20))
        self.assertEqual(result['icawinv'].shape, (32, 15))
        self.assertEqual(result['icaweights'].shape, (15, 32))
        self.assertEqual(result['icasphere'].shape, (32, 32))
        self.assertEqual(len(result['icachansind']), 32)

        self.assertEqual(len(result['chanlocs']), 32)
        for i, chan in enumerate(result['chanlocs']):
            self.assertEqual(chan['labels'], f'EEG{i:03d}')
            self.assertEqual(chan['type'], 'EEG')

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    @XFAIL_178
    def test_data_is_channel_major(self):
        """Epoched data is stored channel-major (nbchan, pnts, trials), matching EEGPrep/EEGLAB."""
        n_channels, n_times, n_epochs = 8, 64, 4
        epochs = _make_epochs(n_channels, n_times, n_epochs)
        ica = ICA(n_components=4, random_state=0)
        ica.fit(epochs)

        result = eeg_mne2eeg_epochs(epochs, ica)

        self.assertEqual(result['data'].shape, (n_channels, n_times, n_epochs))
        # Per-epoch values match MNE's (epoch, channel, time) reshaped to channel-major.
        mne_data = epochs.get_data()
        for trial in range(n_epochs):
            np.testing.assert_allclose(result['data'][:, :, trial], mne_data[trial, :, :])

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    @XFAIL_178
    def test_ica_reconstruction_algebra(self):
        """EEGLAB ICA fields satisfy data ≈ icawinv @ icaact and icaact ≈ icaweights @ icasphere @ data."""
        n_channels, n_times, n_epochs, n_components = 8, 64, 3, 8
        epochs = _make_epochs(n_channels, n_times, n_epochs)
        ica = ICA(n_components=n_components, random_state=0)
        ica.fit(epochs)

        result = eeg_mne2eeg_epochs(epochs, ica)

        # Forward (mixing) reconstruction per epoch.
        for trial in range(n_epochs):
            data_recon = result['icawinv'] @ result['icaact'][:, :, trial]
            np.testing.assert_allclose(data_recon, result['data'][:, :, trial], atol=1e-6)

        # Inverse (unmixing) reconstruction per epoch.
        for trial in range(n_epochs):
            act_recon = result['icaweights'] @ result['icasphere'] @ result['data'][:, :, trial]
            np.testing.assert_allclose(act_recon, result['icaact'][:, :, trial], atol=1e-6)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    @XFAIL_178_STDOUT
    def test_emits_no_stdout(self):
        """User-facing converter must not print to stdout."""
        epochs = _make_epochs()
        ica = ICA(n_components=4, random_state=0)
        ica.fit(epochs)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eeg_mne2eeg_epochs(epochs, ica)
        self.assertEqual(buf.getvalue(), "")


if __name__ == '__main__':
    unittest.main()
