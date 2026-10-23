"""Tests for eeg_checkset eventconsistency mode and pop_epoch boundary detection."""

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, _eventconsistency
from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_select import pop_select


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

    def test_leading_boundary_duration_zero_and_open_unit_latency(self):
        removed = _eventconsistency(
            {
                'event': [
                    {'type': 'Boundary', 'latency': 0.5, 'duration': 0},
                    {'type': 'stim', 'latency': 10, 'duration': 1},
                ],
                'pnts': 100,
                'trials': 1,
            }
        )
        assert [(event['type'], event['latency']) for event in removed['event']] == [('stim', 10)]

        clamped = _eventconsistency(
            {
                'event': [
                    {'type': 'boundary', 'latency': 0.25, 'duration': 2},
                    {'type': 'stim', 'latency': 10, 'duration': 0},
                ],
                'pnts': 100,
                'trials': 1,
            }
        )
        assert [(event['type'], event['latency'], event['duration']) for event in clamped['event']] == [
            ('boundary', 0.5, 2),
            ('stim', 10, 0),
        ]

    def test_duplicate_boundary_run_keeps_every_duration(self):
        EEG = {
            'event': [
                {'type': 'boundary', 'latency': 20, 'duration': 4},
                {'type': 'stim', 'latency': 40, 'duration': 0},
                {'type': 'boundary', 'latency': 20, 'duration': 1},
                {'type': 'boundary', 'latency': 20, 'duration': 2},
                {'type': 'stim', 'latency': 10, 'duration': 0},
            ],
            'pnts': 100,
            'trials': 1,
            'setname': 'test',
        }
        result = _eventconsistency(EEG)
        assert [(event['type'], event['latency'], event['duration']) for event in result['event']] == [
            ('stim', 10, 0),
            ('boundary', 20, 7),
            ('stim', 40, 0),
        ]

    def test_final_boundary_at_even_pnts_plus_one_is_removed(self):
        EEG = {
            'event': [
                {'type': 'boundary', 'latency': 4.5, 'duration': 1},
                {'type': 'boundary', 'latency': 5, 'duration': 3},
                {'type': 'stim', 'latency': 2, 'duration': 0},
            ],
            'pnts': 4,
            'trials': 1,
            'setname': 'test',
        }
        result = _eventconsistency(EEG)
        assert [(event['type'], event['latency'], event['duration']) for event in result['event']] == [
            ('stim', 2, 0),
            ('boundary', 4.5, 1),
        ]

    def test_numeric_boundary_follows_option_boundary99(self):
        EEG = {
            'event': [
                {'type': -99, 'latency': 0.5, 'duration': 0},
                {'type': 'stim', 'latency': 8, 'duration': 0},
            ],
            'pnts': 20,
            'trials': 1,
        }
        previous = EEG_OPTIONS['option_boundary99']
        EEG_OPTIONS['option_boundary99'] = 0
        try:
            kept = _eventconsistency({**EEG, 'event': [dict(event) for event in EEG['event']]})
            assert [event['type'] for event in kept['event']] == [-99, 'stim']
            EEG_OPTIONS['option_boundary99'] = 1
            removed = _eventconsistency(EEG)
            assert [event['type'] for event in removed['event']] == ['stim']
        finally:
            EEG_OPTIONS['option_boundary99'] = previous

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

    def test_duration_is_filled_on_every_event_when_any_event_has_one(self):
        # EEGLAB eeg_checkset.m:564-577 (d0963be9): isfield is true for the
        # whole struct array, then isempty durations become 0.
        filled = _eventconsistency(
            {
                'event': [
                    {'type': 'boundary', 'latency': 50, 'duration': 100},
                    {'type': 'square', 'latency': 10},
                    {'type': 'rt', 'latency': 80, 'duration': ''},
                    {'type': 'stim', 'latency': 90, 'duration': np.array([])},
                ],
                'pnts': 200,
                'trials': 1,
            }
        )
        assert [event['duration'] for event in filled['event']] == [0, 100, 0, 0]

        untouched = _eventconsistency(
            {
                'event': [
                    {'type': 'square', 'latency': 10},
                    {'type': 'rt', 'latency': 80},
                ],
                'pnts': 200,
                'trials': 1,
            }
        )
        assert all('duration' not in event for event in untouched['event'])

    def test_epoched_events_sort_by_epoch_and_fill_duration(self):
        filled = _eventconsistency(
            {
                'event': [
                    {'type': 'stim', 'latency': 10.0, 'epoch': 2, 'duration': 1.0},
                    {'type': 'stim', 'latency': 4.0, 'epoch': 1},
                ],
                'pnts': 20,
                'trials': 2,
            }
        )
        events = list(filled['event'])
        assert [(event['epoch'], event['latency'], event['duration']) for event in events] == [
            (1, 4.0, 0),
            (2, 10.0, 1.0),
        ]

    def test_empty_event_list_stays_empty(self):
        eeg = {'event': [], 'pnts': 10, 'trials': 1}
        _eventconsistency(eeg)
        assert list(np.asarray(eeg['event'], dtype=object).ravel()) == []

    def test_nan_duration_is_not_empty(self):
        # MATLAB isempty(NaN) is false: NaN is a 1-by-1 double, not [].
        filled = _eventconsistency(
            {
                'event': [
                    {'type': 'boundary', 'latency': 5.0, 'duration': 3.0},
                    {'type': 'stim', 'latency': 8.0, 'duration': np.nan},
                ],
                'pnts': 20,
                'trials': 1,
            }
        )
        stim = next(event for event in filled['event'] if event['type'] == 'stim')
        assert np.isnan(stim['duration'])

    def test_rejection_gives_every_event_a_duration(self):
        eeg = pop_loadset('sample_data/eeglab_data.set')

        def durations_of(dataset):
            events = list(np.asarray(dataset['event'], dtype=object).ravel())
            return [event['duration'] for event in events]

        rejected = durations_of(eeg_eegrej(eeg, np.array([[101, 200]])))
        assert rejected[0] == 100
        assert set(rejected[1:]) == {0}
        selected = durations_of(pop_select(eeg, rmtime=[1, 2]))
        assert selected[0] == 129
        assert set(selected[1:]) == {0}
