"""
Test suite for eeg_mne2eeg.py - MNE to EEGLAB conversion.

This module tests the eeg_mne2eeg function that converts MNE objects to EEGLAB datasets.
"""

import unittest
import sys
import numpy as np
import tempfile
import os
import shutil

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

from eegprep.functions.miscfunc.eeg_mne2eeg import eeg_mne2eeg, _mne_events_to_eeglab_events

# Import fixtures - use absolute import to work with unittest discovery
try:
    # Try pytest-style relative import first
    from .fixtures import create_test_eeg
except (ImportError, ValueError):
    # Fallback for unittest: add tests dir to path if needed
    import os

    test_dir = os.path.dirname(os.path.abspath(__file__))
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    from fixtures import create_test_eeg


class TestEEGMNE2EEG(unittest.TestCase):
    """Test cases for eeg_mne2eeg function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_raw_object(self):
        """Test conversion of MNE Raw object."""
        # Create a simple MNE Raw object
        n_channels = 32
        n_times = 1000
        sfreq = 500.0

        # Create channel names
        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]

        # Create MNE info
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')

        # Create random data
        data = np.random.randn(n_channels, n_times)

        # Create Raw object
        raw = mne.io.RawArray(data, info)

        result = eeg_mne2eeg(raw)

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
        self.assertEqual(result['trials'], 1)
        self.assertEqual(result['srate'], sfreq)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_object(self):
        """Test conversion of MNE Epochs object."""
        # Create a simple MNE Epochs object
        n_channels = 32
        n_times = 100
        n_epochs = 10
        sfreq = 500.0

        # Create channel names
        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]

        # Create MNE info
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')

        # Create random data
        data = np.random.randn(n_epochs, n_channels, n_times)

        # Create events
        events = np.array([[i, 0, 1] for i in range(n_epochs)])
        event_id = {'event': 1}

        # Create Epochs object
        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        result = eeg_mne2eeg(epochs)

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

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_with_events(self):
        """Test conversion with MNE events."""
        # Create a simple MNE Epochs object with events
        n_channels = 16
        n_times = 100
        n_epochs = 5
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

        # Create events with different types
        events = np.array(
            [
                [0, 0, 1],  # event type 1
                [1, 0, 2],  # event type 2
                [2, 0, 1],  # event type 1
                [3, 0, 3],  # event type 3
                [4, 0, 2],  # event type 2
            ]
        )
        event_id = {'stimulus': 1, 'response': 2, 'feedback': 3}

        epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

        result = eeg_mne2eeg(epochs)

        # Check that result is a dict
        self.assertIsInstance(result, dict)

        # Check that events were converted
        self.assertIn('event', result)
        self.assertIsInstance(result['event'], list)

        # Check event count
        self.assertEqual(len(result['event']), 5)

        # Check event types
        event_types = [event['type'] for event in result['event']]
        self.assertIn('stimulus', event_types)
        self.assertIn('response', event_types)
        self.assertIn('feedback', event_types)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_no_events(self):
        """Test conversion with no events."""
        # Create MNE Epochs object with no events
        n_channels = 16
        n_times = 100
        n_epochs = 3
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

        # Create events with no event_id mapping
        events = np.array([[i, 0, 999] for i in range(n_epochs)])  # Unknown event type

        epochs = mne.EpochsArray(data, info, events, tmin=0)

        result = eeg_mne2eeg(epochs)

        # Check that result is a dict
        self.assertIsInstance(result, dict)

        # Check that events were converted with string types
        self.assertIn('event', result)
        self.assertIsInstance(result['event'], list)
        self.assertEqual(len(result['event']), 3)

        # Check that event types are strings
        for event in result['event']:
            self.assertIsInstance(event['type'], str)
            self.assertEqual(event['type'], '999')


class TestMNEEventsToEEGLABEvents(unittest.TestCase):
    """Test cases for _mne_events_to_eeglab_events function."""

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_mne_events_to_eeglab_events_annotations(self):
        """Test conversion of MNE annotations to EEGLAB events."""
        # Create MNE annotations
        annotations = mne.Annotations(
            onset=[0.1, 0.5, 1.0], duration=[0.05, 0.05, 0.05], description=['event1', 'event2', 'event3']
        )

        # Create a mock raw object with annotations
        class MockRaw:
            def __init__(self, annotations, sfreq):
                self.annotations = annotations
                self.info = {'sfreq': sfreq}

        raw = MockRaw(annotations, 500.0)

        result = _mne_events_to_eeglab_events(raw)

        # Check result structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

        # Check event structure
        for event in result:
            self.assertIn('latency', event)
            self.assertIn('type', event)
            self.assertIsInstance(event['latency'], int)
            self.assertIsInstance(event['type'], str)

        # Check latency values (1-based indexing)
        latencies = [event['latency'] for event in result]
        expected_latencies = [int(0.1 * 500) + 1, int(0.5 * 500) + 1, int(1.0 * 500) + 1]
        self.assertEqual(latencies, expected_latencies)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_mne_events_to_eeglab_events_events_array(self):
        """Test conversion of MNE events array to EEGLAB events."""
        # Create MNE events array
        events = np.array(
            [
                [0, 0, 1],  # sample 0, event type 1
                [100, 0, 2],  # sample 100, event type 2
                [200, 0, 1],  # sample 200, event type 1
            ]
        )

        # Create a mock epochs object with events and event_id
        class MockEpochs:
            def __init__(self, events, event_id, sfreq):
                self.events = events
                self.event_id = event_id
                self.info = {'sfreq': sfreq}

        event_id = {'stimulus': 1, 'response': 2}
        epochs = MockEpochs(events, event_id, 500.0)

        result = _mne_events_to_eeglab_events(epochs)

        # Check result structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

        # Check event structure
        for event in result:
            self.assertIn('latency', event)
            self.assertIn('type', event)
            self.assertIsInstance(event['latency'], int)
            self.assertIsInstance(event['type'], str)

        # Check latency values (1-based indexing)
        latencies = [event['latency'] for event in result]
        expected_latencies = [1, 101, 201]
        self.assertEqual(latencies, expected_latencies)

        # Check event types
        event_types = [event['type'] for event in result]
        expected_types = ['stimulus', 'response', 'stimulus']
        self.assertEqual(event_types, expected_types)


if __name__ == '__main__':
    unittest.main()
