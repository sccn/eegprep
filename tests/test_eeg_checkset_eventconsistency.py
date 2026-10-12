"""Tests for eeg_checkset eventconsistency mode and pop_epoch boundary detection."""

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, _eventconsistency


class TestEventconsistencyOperations:
    """Unit tests for _eventconsistency helper."""

    def test_removes_out_of_bounds_events(self):
        EEG = {
            'event': [
                {'type': 'stim', 'latency': 0.3},  # < 0.5 -> remove
                {'type': 'stim', 'latency': 0.5},  # == 0.5 -> keep
                {'type': 'stim', 'latency': 500},  # ok
                {'type': 'stim', 'latency': 1001},  # == pnts*trials+1 -> keep
                {'type': 'stim', 'latency': 1002},  # > pnts*trials+1 -> remove
            ],
            'pnts': 1000,
            'trials': 1,
        }
        result = _eventconsistency(EEG)
        lats = [e['latency'] for e in result['event']]
        assert 0.3 not in lats
        assert 1002 not in lats
        assert 0.5 in lats
        assert 500 in lats
        assert 1001 in lats

    def test_removes_invalid_epoch_numbers(self):
        EEG = {
            'event': [
                {'type': 'stim', 'latency': 10, 'epoch': 1},
                {'type': 'stim', 'latency': 20, 'epoch': 0},  # invalid
                {'type': 'stim', 'latency': 30, 'epoch': 3},
                {'type': 'stim', 'latency': 40, 'epoch': 4},  # invalid (> trials)
            ],
            'pnts': 100,
            'trials': 3,
        }
        result = _eventconsistency(EEG)
        epochs = [e['epoch'] for e in result['event']]
        assert 0 not in epochs
        assert 4 not in epochs
        assert 1 in epochs
        assert 3 in epochs

    def test_removes_duplicate_boundary_events(self):
        EEG = {
            'event': [
                {'type': 'stim', 'latency': 50},
                {'type': 'boundary', 'latency': 100.5, 'duration': 30},
                {'type': 'boundary', 'latency': 100.5, 'duration': 20},  # duplicate
                {'type': 'stim', 'latency': 200},
            ],
            'pnts': 1000,
            'trials': 1,
            'setname': 'test',
        }
        result = _eventconsistency(EEG)
        boundaries = [e for e in result['event'] if e['type'] == 'boundary']
        assert len(boundaries) == 1
        assert boundaries[0]['duration'] == 50  # merged: 30 + 20

    def test_resorts_events_by_epoch_latency(self):
        EEG = {
            'event': [
                {'type': 'b', 'latency': 200, 'epoch': 2},
                {'type': 'a', 'latency': 100, 'epoch': 1},
                {'type': 'c', 'latency': 150, 'epoch': 1},
            ],
            'pnts': 300,
            'trials': 2,
        }
        result = _eventconsistency(EEG)
        lats = [(e['epoch'], e['latency']) for e in result['event']]
        assert lats == [(1, 100), (1, 150), (2, 200)]

    def test_first_boundary_duration_lt_1_removed(self):
        EEG = {
            'event': [
                {'type': 'boundary', 'latency': 0.5, 'duration': 0.5},
                {'type': 'stim', 'latency': 100},
            ],
            'pnts': 1000,
            'trials': 1,
        }
        result = _eventconsistency(EEG)
        assert result['event'][0]['type'] == 'stim'

    def test_eeg_checkset_accepts_eventconsistency_arg(self):
        """Verify eeg_checkset('eventconsistency') calls _eventconsistency."""
        EEG = {
            'data': np.random.randn(2, 200).astype(np.float32),
            'srate': 100.0,
            'nbchan': 2,
            'pnts': 200,
            'trials': 1,
            'xmin': 0.0,
            'xmax': 1.99,
            'event': [
                {'type': 'stim', 'latency': float('nan')},
                {'type': 'stim', 'latency': 100},
            ],
            'epoch': [],
            'saved': 'no',
        }
        result = eeg_checkset(EEG, 'eventconsistency')
        # NaN event should have been removed
        assert len(result['event']) == 1
