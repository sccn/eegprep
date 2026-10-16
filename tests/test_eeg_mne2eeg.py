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
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

from eegprep.functions.miscfunc._eeg_mne import _subject_text
from eegprep.functions.miscfunc.eeg_eeg2mne import eeg_eeg2mne
from eegprep.functions.miscfunc.eeg_mne2eeg import eeg_mne2eeg, _mne_events_to_eeglab_events
from eegprep.functions.popfunc.pop_loadset import pop_loadset

REPO_ROOT = Path(__file__).resolve().parents[1]
EPOCHED_DATASET = REPO_ROOT / 'sample_data' / 'eeglab_data_epochs_ica.set'
CONTINUOUS_DATASET = REPO_ROOT / 'sample_data' / 'eeglab_data.set'

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
        n_channels = 32
        n_times = 1000
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]

        info = mne.create_info(ch_names, sfreq, ch_types='eeg')

        data = np.random.randn(n_channels, n_times)

        raw = mne.io.RawArray(data, info)

        result = eeg_mne2eeg(raw)

        # Check that result is a dict (EEGLAB format)
        self.assertIsInstance(result, dict)

        self.assertIn('data', result)
        self.assertIn('srate', result)
        self.assertIn('nbchan', result)
        self.assertIn('pnts', result)
        self.assertIn('trials', result)

        self.assertEqual(result['nbchan'], n_channels)
        self.assertEqual(result['pnts'], n_times)
        self.assertEqual(result['trials'], 1)
        self.assertEqual(result['srate'], sfreq)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_epochs_object(self):
        """Test conversion of MNE Epochs object."""
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

        result = eeg_mne2eeg(epochs)

        # Check that result is a dict (EEGLAB format)
        self.assertIsInstance(result, dict)

        self.assertIn('data', result)
        self.assertIn('srate', result)
        self.assertIn('nbchan', result)
        self.assertIn('pnts', result)
        self.assertIn('trials', result)

        self.assertEqual(result['nbchan'], n_channels)
        self.assertEqual(result['pnts'], n_times)
        self.assertEqual(result['trials'], n_epochs)
        self.assertEqual(result['srate'], sfreq)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_with_events(self):
        """Test conversion with MNE events."""
        n_channels = 16
        n_times = 100
        n_epochs = 5
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

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

        self.assertIsInstance(result, dict)

        self.assertIn('event', result)
        self.assertIsInstance(result['event'], (list, np.ndarray))

        self.assertEqual(len(result['event']), 5)

        event_types = [event['type'] for event in result['event']]
        self.assertIn('stimulus', event_types)
        self.assertIn('response', event_types)
        self.assertIn('feedback', event_types)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_mne2eeg_no_events(self):
        """Test conversion with no events."""
        n_channels = 16
        n_times = 100
        n_epochs = 3
        sfreq = 500.0

        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        data = np.random.randn(n_epochs, n_channels, n_times)

        events = np.array([[i, 0, 999] for i in range(n_epochs)])  # Unknown event type

        epochs = mne.EpochsArray(data, info, events, tmin=0)

        result = eeg_mne2eeg(epochs)

        self.assertIsInstance(result, dict)

        self.assertIn('event', result)
        self.assertIsInstance(result['event'], (list, np.ndarray))
        self.assertEqual(len(result['event']), 3)

        for event in result['event']:
            self.assertIsInstance(event['type'], int)
            self.assertEqual(event['type'], 999)


def _event_rows(events):
    return [
        (
            event['type'],
            event['latency'],
            int(event['epoch']),
            int(event['position']),
        )
        for event in events
    ]


def _epoch_rows(epochs):
    rows = []
    for epoch in epochs:
        rows.append(
            (
                [int(value) for value in epoch['event']],
                [str(value) for value in epoch['eventtype']],
                [float(value) for value in epoch['eventlatency']],
                [int(value) for value in epoch['eventposition']],
            )
        )
    return rows


def _mne_rows(epochs):
    subject = (epochs.info.get('subject_info') or {}).get('his_id')
    names = {code: name for name, code in epochs.event_id.items()}
    locking = [names[int(code)] for code in epochs.events[:, 2]]
    per_epoch = epochs.get_annotations_per_epoch(with_extras=True)
    markers = []
    for epoch_annots in per_epoch:
        markers.append(
            tuple(
                (str(description), extras.get('latency'), int(extras.get('epoch')))
                for _onset, _duration, description, extras in epoch_annots
            )
        )
    return subject, tuple(locking), tuple(markers)


class TestMNERoundTrip(unittest.TestCase):
    """EEG and MNE epoch conversions preserve events and subject."""

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_sample_epochs_round_trip_events_and_subject(self):
        original = pop_loadset(str(EPOCHED_DATASET))
        original['subject'] = 'sub-01'
        expected_events = _event_rows(original['event'])
        expected_epochs = _epoch_rows(original['epoch'])

        as_mne = eeg_eeg2mne(original)
        restored = eeg_mne2eeg(as_mne)
        self.assertEqual(restored['subject'], 'sub-01')
        self.assertEqual(_event_rows(restored['event']), expected_events)
        self.assertEqual(_epoch_rows(restored['epoch']), expected_epochs)
        for original_event, restored_event in zip(original['event'], restored['event']):
            self.assertNotIn('urevent', restored_event)
            self.assertEqual(set(restored_event), set(original_event) - {'urevent'})
        for original_epoch, restored_epoch in zip(original['epoch'], restored['epoch']):
            self.assertNotIn('eventurevent', restored_epoch)
            self.assertEqual(set(restored_epoch), set(original_epoch) - {'eventurevent'})

        as_mne_again = eeg_eeg2mne(restored)
        self.assertEqual(_mne_rows(as_mne_again), _mne_rows(as_mne))

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_sample_continuous_round_trip_subject(self):
        original = pop_loadset(str(CONTINUOUS_DATASET))
        original['subject'] = 'sub-02'
        as_mne = eeg_eeg2mne(original)
        self.assertEqual(as_mne.info['subject_info']['his_id'], 'sub-02')
        restored = eeg_mne2eeg(as_mne)
        self.assertEqual(restored['subject'], 'sub-02')
        as_mne_again = eeg_eeg2mne(restored)
        self.assertEqual(as_mne_again.info['subject_info']['his_id'], 'sub-02')

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_half_open_window_assigns_boundary_event_once(self):
        sfreq = 100.0
        n_times = 71
        tmin = -0.2
        info = mne.create_info(['EEG000'], sfreq, ch_types='eeg')
        data = np.zeros((2, 1, n_times))
        events = np.array([[0, 0, 1], [n_times, 0, 1]])
        epochs = mne.EpochsArray(data, info, events, tmin=tmin, event_id={'target': 1}, verbose=False)
        epochs.set_annotations(
            mne.Annotations(
                onset=[0.505, 0.51],
                duration=[0.0, 0.0],
                description=['edge', 'next'],
            )
        )

        result = eeg_mne2eeg(epochs)

        got = [(event['type'], int(event['epoch'])) for event in result['event']]
        self.assertEqual(got.count(('edge', 1)), 1)
        self.assertEqual(got.count(('next', 2)), 1)
        self.assertNotIn(('edge', 2), got)
        self.assertNotIn(('next', 1), got)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_numeric_types_and_duration_round_trip(self):
        eeg = create_test_eeg(n_channels=2, n_samples=50, srate=128.0, n_trials=2)
        eeg['event'] = [
            {'type': 1, 'latency': 1, 'epoch': 1, 'duration': 16, 'urevent': 3},
            {'type': 2, 'latency': 51, 'epoch': 2, 'duration': 0, 'urevent': 4},
        ]
        eeg['urevent'] = []
        eeg['subject'] = 'sub-01'

        restored = eeg_mne2eeg(eeg_eeg2mne(eeg))

        self.assertEqual([event['type'] for event in restored['event']], [1, 2])
        self.assertTrue(all(type(event['type']) is int for event in restored['event']))
        self.assertEqual(restored['event'][0]['duration'], 16)
        self.assertNotIn('duration', restored['event'][1])
        self.assertTrue(all('urevent' not in event for event in restored['event']))

        mixed = create_test_eeg(n_channels=2, n_samples=50, srate=128.0, n_trials=2)
        mixed['event'] = [
            {'type': 1, 'latency': 1, 'epoch': 1},
            {'type': 'square', 'latency': 51, 'epoch': 2},
        ]
        mixed['urevent'] = []
        mixed_restored = eeg_mne2eeg(eeg_eeg2mne(mixed))
        self.assertEqual([event['type'] for event in mixed_restored['event']], ['1', 'square'])
        self.assertTrue(all(type(event['type']) is str for event in mixed_restored['event']))

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_event_id_keys_are_plain_str(self):
        eeg = create_test_eeg(n_channels=2, n_samples=50, srate=128.0, n_trials=2)
        eeg['event'][0]['type'] = np.str_('square')
        eeg['event'][1]['type'] = np.str_('rt')
        converted = eeg_eeg2mne(eeg)
        self.assertTrue(all(type(name) is str for name in converted.event_id))

        info = mne.create_info(['EEG000'], 100.0, ch_types='eeg')
        data = np.zeros((1, 1, 10))
        epochs = mne.EpochsArray(
            data,
            info,
            np.array([[0, 0, 1]]),
            tmin=0.0,
            event_id={np.str_('stimulus'): 1},
            verbose=False,
        )
        restored = eeg_mne2eeg(epochs)
        self.assertTrue(all(type(event['type']) is str for event in restored['event']))
        self.assertEqual(restored['event'][0]['type'], 'stimulus')

    def test_subject_text_rejects_multiple_values(self):
        self.assertEqual(_subject_text(np.array([])), '')
        with self.assertRaises(ValueError):
            _subject_text(np.array(['a', 'b']))


class TestMNEEventsToEEGLABEvents(unittest.TestCase):
    """Test cases for _mne_events_to_eeglab_events function."""

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_mne_events_to_eeglab_events_annotations(self):
        """Test conversion of MNE annotations to EEGLAB events."""
        annotations = mne.Annotations(
            onset=[0.1, 0.5, 1.0], duration=[0.05, 0.05, 0.05], description=['event1', 'event2', 'event3']
        )

        class MockRaw:
            def __init__(self, annotations, sfreq):
                self.annotations = annotations
                self.info = {'sfreq': sfreq}

        raw = MockRaw(annotations, 500.0)

        result = _mne_events_to_eeglab_events(raw)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

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
        events = np.array(
            [
                [0, 0, 1],  # sample 0, event type 1
                [100, 0, 2],  # sample 100, event type 2
                [200, 0, 1],  # sample 200, event type 1
            ]
        )

        class MockEpochs:
            def __init__(self, events, event_id, sfreq):
                self.events = events
                self.event_id = event_id
                self.info = {'sfreq': sfreq}

        event_id = {'stimulus': 1, 'response': 2}
        epochs = MockEpochs(events, event_id, 500.0)

        result = _mne_events_to_eeglab_events(epochs)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

        for event in result:
            self.assertIn('latency', event)
            self.assertIn('type', event)
            self.assertIsInstance(event['latency'], int)
            self.assertIsInstance(event['type'], str)

        # Check latency values (1-based indexing)
        latencies = [event['latency'] for event in result]
        expected_latencies = [1, 101, 201]
        self.assertEqual(latencies, expected_latencies)

        event_types = [event['type'] for event in result]
        expected_types = ['stimulus', 'response', 'stimulus']
        self.assertEqual(event_types, expected_types)


if __name__ == '__main__':
    unittest.main()
