# test_pop_select.py
import os
import unittest
import math
from pathlib import Path
import numpy as np
import copy
import pytest
import scipy.io

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from tests.eeg_options_snapshot import eeg_options_snapshot, restore_eeg_options
from tests.fixtures import SAMPLE_DATASET_PATH

# where the test resources
web_root = 'https://sccntestdatasets.s3.us-east-2.amazonaws.com/'
local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')


def ensure_file(fname: str) -> str:  # duplicate of test_clean_rawdata.py
    """Download a file if it does not exist and return the local path."""
    full_url = f"{web_root}{fname}"
    local_file = os.path.abspath(f"{local_url}{fname}")
    if not os.path.exists(local_file):
        from urllib.request import urlretrieve

        urlretrieve(full_url, local_file)
    return local_file


def _chan_labels(EEG):
    labs = []
    if EEG['chanlocs'] is not None and len(EEG['chanlocs']) > 0:
        for ch in EEG['chanlocs']:
            # EEGLAB uses 'labels'
            labs.append(ch.get('labels') or ch.get('label') or '')
    return labs


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestPopSelectParity(unittest.TestCase):
    def setUp(self):
        self.EEG_py = pop_loadset(ensure_file('FlankerTest.set'))
        self.eeglab = get_eeglab('MAT')  # MATLAB bridge

    def test_parity_trial_subset(self):
        EEG_py_epochs, _ = pop_epoch(self.EEG_py, 'S 11', [-0.2, 0.2])

        trials = int(EEG_py_epochs['trials'])
        k = min(5, trials)
        keep_trials = list(range(1, k + 1))  # 1-based

        EEG_py_out = pop_select(copy.deepcopy(EEG_py_epochs), trial=keep_trials)
        EEG_mat_out = self.eeglab.pop_select(copy.deepcopy(EEG_py_epochs), 'trial', keep_trials)

        self.assertEqual(EEG_py_out['trials'], k)
        self.assertEqual(EEG_mat_out['trials'], k)
        self.assertEqual(EEG_py_out['nbchan'], EEG_mat_out['nbchan'])
        self.assertEqual(EEG_py_out['pnts'], EEG_mat_out['pnts'])
        self.assertTrue(np.allclose(EEG_py_out['data'], EEG_mat_out['data'], atol=1e-7, equal_nan=True))

        # If events exist, ensure counts match
        if EEG_py_out.get('event') is not None and EEG_mat_out.get('event') is not None:
            self.assertEqual(len(EEG_py_out['event']), len(EEG_mat_out['event']))


def _boundary_events(EEG):
    """Return (latency, duration) pairs of boundary events, in order."""
    return [(float(ev['latency']), float(ev['duration'])) for ev in EEG['event'] if str(ev.get('type')) == 'boundary']


class TestPopSelectContinuousRemoval(unittest.TestCase):
    """Sample-exact removal/keep semantics on continuous data.

    Expected values were confirmed against EEGLAB pop_select on
    sample_data/eeglab_data.set (30504 samples, 128 Hz). Point ranges are
    1-based and inclusive at both ends; a boundary event sits half a sample
    before the first removed sample with duration equal to the removed count.
    """

    def setUp(self):
        self.EEG = pop_loadset(str(SAMPLE_DATASET_PATH))
        self.assertEqual(self.EEG['pnts'], 30504)

    def _check(self, EEG_out, pnts, boundaries):
        self.assertEqual(EEG_out['pnts'], pnts)
        self.assertEqual(EEG_out['data'].shape[1], pnts)
        self.assertEqual(_boundary_events(EEG_out), boundaries)

    def test_nopoint_removes_inclusive_range(self):
        for key in ('nopoint', 'rmpoint'):
            EEG_out = pop_select(copy.deepcopy(self.EEG), **{key: [276, 525]})
            self._check(EEG_out, 30254, [(275.5, 250.0)])

    def test_nopoint_first_and_last_samples(self):
        EEG_out = pop_select(copy.deepcopy(self.EEG), nopoint=[1, 10])
        self._check(EEG_out, 30494, [(0.5, 10.0)])
        np.testing.assert_array_equal(EEG_out['data'], self.EEG['data'][:, 10:])

        EEG_out = pop_select(copy.deepcopy(self.EEG), nopoint=[30495, 30504])
        self._check(EEG_out, 30494, [(30494.5, 10.0)])
        np.testing.assert_array_equal(EEG_out['data'], self.EEG['data'][:, :30494])

    def test_nopoint_two_regions(self):
        EEG_out = pop_select(copy.deepcopy(self.EEG), nopoint=[[100, 200], [300, 400]])
        self._check(EEG_out, 30302, [(99.5, 101.0), (198.5, 101.0)])

    def test_notime_removes_inclusive_range(self):
        for key in ('notime', 'rmtime'):
            EEG_out = pop_select(copy.deepcopy(self.EEG), **{key: [2.15, 4.1]})
            self._check(EEG_out, 30253, [(275.5, 251.0)])

    def test_point_and_time_keep_range(self):
        EEG_out = pop_select(copy.deepcopy(self.EEG), point=[276, 525])
        self._check(EEG_out, 250, [(0.5, 275.0), (250.5, 29979.0)])

        EEG_out = pop_select(copy.deepcopy(self.EEG), time=[2.15, 4.1])
        self._check(EEG_out, 251, [(0.5, 275.0), (251.5, 29978.0)])


class TestPopSelectFunctional(unittest.TestCase):
    def setUp(self):
        self.EEG = pop_loadset(ensure_file('FlankerTest.set'))

    def test_nochannel_records_removed(self):
        labels = _chan_labels(self.EEG)
        if len(labels) < 4:
            self.skipTest("Need at least 4 channels to test removal bookkeeping")

        drop = labels[2:4]
        EEG_out = pop_select(copy.deepcopy(self.EEG), nochannel=drop)

        self.assertEqual(EEG_out['nbchan'], len(labels) - 2)
        # removedchans bookkeeping
        chaninfo = EEG_out.get('chaninfo', {})
        removed = chaninfo.get('removedchans', [])
        self.assertGreaterEqual(len(removed), 2)

    def test_caller_eeg_is_not_mutated(self):
        EEG = copy.deepcopy(self.EEG)
        original_data = EEG['data'].copy()
        original_nbchan = int(EEG['nbchan'])
        events = EEG.get('event')
        original_event_count = 0 if events is None else len(events)
        original_first_latency = (
            float(events[0]['latency']) if original_event_count and 'latency' in events[0] else None
        )

        labels = _chan_labels(EEG)
        self.assertGreaterEqual(len(labels), 2, "Dataset must have at least 2 channels")
        EEG_out = pop_select(EEG, channel=labels[:2])

        # The returned dataset reflects the selection ...
        self.assertEqual(EEG_out['nbchan'], 2)
        # ... while the caller's EEG is left completely untouched.
        self.assertTrue(np.array_equal(original_data, EEG['data']))
        self.assertEqual(int(EEG['nbchan']), original_nbchan)
        events_after = EEG.get('event')
        self.assertEqual(0 if events_after is None else len(events_after), original_event_count)
        if original_first_latency is not None:
            self.assertEqual(float(events_after[0]['latency']), original_first_latency)


class TestPopSelectEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions in pop_select."""

    def setUp(self):
        """Create a minimal test EEG structure."""
        self.EEG = {
            'data': np.random.randn(4, 100, 3),  # 4 channels, 100 points, 3 trials
            'nbchan': 4,
            'pnts': 100,
            'trials': 3,
            'srate': 250.0,
            'xmin': -0.2,
            'xmax': 0.2,
            'chanlocs': [
                {'labels': 'Fz', 'X': 0, 'Y': 1, 'Z': 0},
                {'labels': 'Cz', 'X': 0, 'Y': 0, 'Z': 1},
                {'labels': 'Pz', 'X': 0, 'Y': -1, 'Z': 0},
                {'labels': 'Oz', 'X': 0, 'Y': -1, 'Z': -1},
            ],
            'event': [
                {'type': 'stimulus', 'latency': 50, 'epoch': 1},
                {'type': 'response', 'latency': 150, 'epoch': 2},
                {'type': 'stimulus', 'latency': 250, 'epoch': 3},
            ],
            'epoch': [{}, {}, {}],
            'icaact': None,
            'icawinv': None,
            'icaweights': None,
            'icasphere': None,
            'icachansind': None,
            'specdata': None,
            'specicaact': None,
            'reject': {},
            'stats': {},
            'dipfit': None,
            'roi': None,
            'chaninfo': {},
        }

    def test_channel_selection_by_indices(self):
        """Test channel selection by integer indices."""
        EEG = copy.deepcopy(self.EEG)

        # Select channels by 0-based indices
        EEG_out = pop_select(EEG, channel=[0, 2])  # Fz and Pz

        self.assertEqual(EEG_out['nbchan'], 2)
        self.assertEqual(len(EEG_out['chanlocs']), 2)
        self.assertEqual(EEG_out['chanlocs'][0]['labels'], 'Fz')
        self.assertEqual(EEG_out['chanlocs'][1]['labels'], 'Pz')
        self.assertEqual(EEG_out['data'].shape[0], 2)

    def test_channel_selection_by_type(self):
        """Selecting by chantype keeps only channels whose type matches (no unpack crash)."""
        EEG = copy.deepcopy(self.EEG)
        types = ['EEG', 'EEG', 'EOG', 'EOG']
        for chan, ctype in zip(EEG['chanlocs'], types):
            chan['type'] = ctype

        EEG_out = pop_select(EEG, chantype=['EEG'])

        self.assertEqual(EEG_out['nbchan'], 2)
        self.assertEqual([chan['labels'] for chan in EEG_out['chanlocs']], ['Fz', 'Cz'])

    def test_channel_removal_by_type(self):
        """Removing by rmchantype drops channels whose type matches (no unpack crash)."""
        EEG = copy.deepcopy(self.EEG)
        types = ['EEG', 'EEG', 'EOG', 'EOG']
        for chan, ctype in zip(EEG['chanlocs'], types):
            chan['type'] = ctype

        EEG_out = pop_select(EEG, rmchantype=['EOG'])

        self.assertEqual(EEG_out['nbchan'], 2)
        self.assertEqual([chan['labels'] for chan in EEG_out['chanlocs']], ['Fz', 'Cz'])

    def test_trial_selection_with_events(self):
        """Test trial selection updates events correctly."""
        EEG = copy.deepcopy(self.EEG)

        EEG_out = pop_select(EEG, trial=[1, 2])

        self.assertEqual(EEG_out['trials'], 2)
        # Should have 2 events (from epochs 1 and 2)
        self.assertEqual(len(EEG_out['event']), 2)
        self.assertEqual(EEG_out['event'][0]['epoch'], 1)
        self.assertEqual(EEG_out['event'][1]['epoch'], 2)

    def test_dipfit_removal_warning(self):
        """Test that dipfit is removed when channels are removed."""
        EEG = copy.deepcopy(self.EEG)
        EEG['dipfit'] = [{'some': 'dipole_info'}]
        EEG['roi'] = [{'some': 'roi_info'}]

        EEG_out = pop_select(EEG, channel=[0, 1, 2])  # Remove channel 3

        # dipfit and roi should be cleared
        self.assertEqual(np.asarray(EEG_out['dipfit']).size, 0)
        self.assertEqual(EEG_out['roi'], {})

    def test_single_trial_epoch_field_removal(self):
        """Test that epoch fields are removed from events when only one trial remains."""
        EEG = copy.deepcopy(self.EEG)

        EEG_out = pop_select(EEG, trial=[1])

        self.assertEqual(EEG_out['trials'], 1)
        # epoch field should be removed from events
        for event in EEG_out['event']:
            self.assertNotIn('epoch', event)
        # epoch list should be empty
        self.assertEqual(np.asarray(EEG_out['epoch']).size, 0)


def _dataset(events, *, trials=1, pnts=100):
    data = np.arange(2 * pnts * trials, dtype=np.float64).reshape(2, pnts, trials)
    if trials == 1:
        data = data[:, :, 0]
    srate = 100.0
    xmin = 0.0 if trials == 1 else -0.1
    xmax = xmin + (pnts - 1) / srate
    return {
        'setname': 'eventconsistency',
        'data': data,
        'nbchan': 2,
        'pnts': pnts,
        'trials': trials,
        'srate': srate,
        'xmin': xmin,
        'xmax': xmax,
        'times': np.linspace(xmin * 1000, xmax * 1000, pnts),
        'chanlocs': [
            {'labels': 'Fz', 'type': 'EEG'},
            {'labels': 'Cz', 'type': 'EOG'},
        ],
        'event': events,
        'epoch': [],
        'chaninfo': {},
        'reject': {},
        'stats': {},
    }


def _event_rows(eeg):
    events = eeg.get('event')
    if events is None:
        return []
    if isinstance(events, np.ndarray):
        events = [] if events.size == 0 else list(events)
    rows = []
    for event in events:
        event_type = event.get('type', '')
        if isinstance(event_type, bytes):
            event_type = event_type.decode()
        duration = event.get('duration', 0)
        rows.append((str(event_type), float(event['latency']), float(0 if duration is None else duration)))
    return rows


class TestPopSelectEventConsistency(unittest.TestCase):
    def test_leading_boundary_with_duration_zero_is_removed(self):
        eeg = _dataset(
            [
                {'type': 'Boundary', 'latency': 0.5, 'duration': 0},
                {'type': 'stim', 'latency': 10, 'duration': 1},
            ]
        )
        out = pop_select(eeg, channel=['Fz'])
        self.assertEqual(_event_rows(out), [('stim', 10.0, 1.0)])
        np.testing.assert_array_equal(out['data'], eeg['data'][0:1])

    def test_leading_boundary_latency_in_open_unit_interval_is_clamped(self):
        eeg = _dataset(
            [
                {'type': 'boundary', 'latency': 0.25, 'duration': 2},
                {'type': 'stim', 'latency': 10, 'duration': 0},
            ]
        )
        out = pop_select(eeg, channel=['Fz'])
        self.assertEqual(_event_rows(out), [('boundary', 0.5, 2.0), ('stim', 10.0, 0.0)])

    def test_duplicate_boundaries_merge_every_duration_and_sort(self):
        eeg = _dataset(
            [
                {'type': 'boundary', 'latency': 20, 'duration': 4},
                {'type': 'stim', 'latency': 40, 'duration': 0},
                {'type': 'boundary', 'latency': 20, 'duration': 1},
                {'type': 'boundary', 'latency': 20, 'duration': 2},
                {'type': 'stim', 'latency': 10, 'duration': 0},
            ]
        )
        out = pop_select(eeg, channel=['Fz'])
        self.assertEqual(
            _event_rows(out),
            [('stim', 10.0, 0.0), ('boundary', 20.0, 7.0), ('stim', 40.0, 0.0)],
        )

    def test_out_of_range_epoch_numbers_are_removed(self):
        eeg = _dataset(
            [
                {'type': 'late', 'latency': 30, 'epoch': 4, 'duration': 0},
                {'type': 'second', 'latency': 5, 'epoch': np.int64(2), 'duration': 0},
                {'type': 'bad', 'latency': 12, 'epoch': 0, 'duration': 0},
                {'type': 'first', 'latency': 30, 'epoch': 1, 'duration': 0},
            ],
            trials=3,
            pnts=20,
        )
        out = pop_select(eeg, channel=['Fz'])
        self.assertEqual(_event_rows(out), [('first', 30.0, 0.0), ('second', 5.0, 0.0)])
        np.testing.assert_array_equal(out['data'], eeg['data'][0:1])

    def test_numeric_boundary_type_follows_option_boundary99(self):
        eeg = _dataset(
            [
                {'type': -99, 'latency': 0.5, 'duration': 0},
                {'type': 'stim', 'latency': 8, 'duration': 0},
            ],
            pnts=20,
        )
        previous = EEG_OPTIONS['option_boundary99']
        EEG_OPTIONS['option_boundary99'] = 1
        try:
            out = pop_select(eeg, channel=['Fz'])
        finally:
            EEG_OPTIONS['option_boundary99'] = previous
        self.assertEqual(_event_rows(out), [('stim', 8.0, 0.0)])

    def test_numeric_boundary_duplicates_are_merged(self):
        eeg = _dataset(
            [
                {'type': -99, 'latency': 20, 'duration': 1},
                {'type': -99, 'latency': 20, 'duration': 2},
            ]
        )
        previous = EEG_OPTIONS['option_boundary99']
        EEG_OPTIONS['option_boundary99'] = 1
        try:
            out = pop_select(eeg, channel=['Fz'])
        finally:
            EEG_OPTIONS['option_boundary99'] = previous
        self.assertEqual(_event_rows(out), [('-99', 20.0, 3.0)])

    def test_boundary_runs_chain_only_consecutive_equal_latencies(self):
        eeg = _dataset(
            [
                {'type': 'boundary', 'latency': 20, 'duration': 4},
                {'type': 'stim', 'latency': 10, 'duration': 0},
                {'type': 'boundary', 'latency': 20, 'duration': 1},
                {'type': 'boundary', 'latency': 30, 'duration': 8},
                {'type': 'boundary', 'latency': 20, 'duration': 2},
            ]
        )
        out = pop_select(eeg, channel=['Fz'])
        self.assertEqual(
            _event_rows(out),
            [
                ('stim', 10.0, 0.0),
                ('boundary', 20.0, 5.0),
                ('boundary', 20.0, 2.0),
                ('boundary', 30.0, 8.0),
            ],
        )

    def test_epoched_events_sort_by_epoch_then_latency(self):
        crossed = _dataset(
            [
                {'type': 'A', 'latency': 75, 'epoch': 2, 'duration': 0},
                {'type': 'B', 'latency': 15, 'epoch': 1, 'duration': 0},
                {'type': 'C', 'latency': 5, 'epoch': 1, 'duration': 0},
                {'type': 'D', 'latency': 145, 'epoch': 3, 'duration': 0},
            ],
            trials=3,
            pnts=50,
        )
        same_epoch = _dataset(
            [
                {'type': 'X', 'latency': 85, 'epoch': 2, 'duration': 0},
                {'type': 'Y', 'latency': 72, 'epoch': 2, 'duration': 0},
                {'type': 'Z', 'latency': 8, 'epoch': 1, 'duration': 0},
            ],
            trials=3,
            pnts=50,
        )
        self.assertEqual(
            _epoch_rows(pop_select(crossed, channel=['Fz'])),
            [('C', 5.0, 1.0), ('B', 15.0, 1.0), ('A', 75.0, 2.0), ('D', 145.0, 3.0)],
        )
        self.assertEqual(
            _epoch_rows(pop_select(same_epoch, channel=['Fz'])),
            [('Z', 8.0, 1.0), ('Y', 72.0, 2.0), ('X', 85.0, 2.0)],
        )


def _epoch_rows(eeg):
    events = eeg.get('event')
    if isinstance(events, np.ndarray):
        events = [] if events.size == 0 else list(events)
    return [(str(event['type']), float(event['latency']), float(event['epoch'])) for event in events or []]


def _eeg_channel_labels(eeg):
    locations = eeg.get('chanlocs')
    if isinstance(locations, np.ndarray):
        locations = [] if locations.size == 0 else list(locations)
    labels = []
    for location in locations or []:
        channel_type = location.get('type')
        if isinstance(channel_type, bytes):
            channel_type = channel_type.decode()
        if str(channel_type) == 'EEG':
            labels.append(str(location.get('labels', '')))
    return labels


def _exact_number(left, right):
    left = float(left)
    right = float(right)
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return left == right


def _matlab_strings(value):
    strings = []
    for item in np.asarray(value, dtype=object).ravel():
        if isinstance(item, np.ndarray):
            if item.dtype.kind in 'SU':
                strings.append(''.join(item.astype(str).ravel()))
            elif item.size == 0:
                strings.append('')
            else:
                strings.append(str(item.reshape(-1)[0]))
        elif isinstance(item, bytes):
            strings.append(item.decode())
        else:
            strings.append(str(item))
    return strings


def _matlab_literal(path):
    return str(path).replace('\\', '/').replace("'", "''")


def _matlab_pop_select(engine, set_path, labels, mat_path):
    channel = ', '.join("'" + label.replace("'", "''") + "'" for label in labels)
    engine.eval(
        "EEG = pop_loadset('filename', '" + _matlab_literal(set_path) + "'); "
        "EEG = pop_select(EEG, 'channel', {" + channel + "}); "
        "n = numel(EEG.event); "
        "event_type = cell(1, n); "
        "event_latency = nan(1, n); "
        "event_duration = nan(1, n); "
        "event_epoch = nan(1, n); "
        "has_epoch = isfield(EEG.event, 'epoch'); "
        "for iEvent = 1:n, "
        "event_type{iEvent} = char(string(EEG.event(iEvent).type)); "
        "event_latency(iEvent) = double(EEG.event(iEvent).latency); "
        "if isfield(EEG.event, 'duration') && ~isempty(EEG.event(iEvent).duration), "
        "event_duration(iEvent) = double(EEG.event(iEvent).duration); "
        "end; "
        "if has_epoch && ~isempty(EEG.event(iEvent).epoch), "
        "event_epoch(iEvent) = double(EEG.event(iEvent).epoch); "
        "end; "
        "end; "
        "data = double(EEG.data); "
        "save('"
        + _matlab_literal(mat_path)
        + "', 'event_type', 'event_latency', 'event_duration', 'event_epoch', 'data', '-v7'); "
        "clear EEG data event_type event_latency event_duration event_epoch;",
        nargout=0,
    )
    return scipy.io.loadmat(mat_path, squeeze_me=False)


def _assert_matlab_match(python_eeg, saved):
    python_rows = _event_rows(python_eeg)
    matlab_types = _matlab_strings(saved['event_type'])
    matlab_latency = np.asarray(saved['event_latency'], dtype=float).ravel()
    matlab_duration = np.asarray(saved['event_duration'], dtype=float).ravel()
    assert len(python_rows) == len(matlab_types) == matlab_latency.size == matlab_duration.size, (
        f'{len(python_rows)} Python events vs {len(matlab_types)} MATLAB events'
    )
    for index, (event_type, latency, duration) in enumerate(python_rows):
        assert event_type == matlab_types[index], f'event {index} type {event_type} vs {matlab_types[index]}'
        assert _exact_number(latency, matlab_latency[index]), (
            f'event {index} latency {latency} vs {matlab_latency[index]}'
        )
        assert _exact_number(duration, matlab_duration[index]), (
            f'event {index} duration {duration} vs {matlab_duration[index]}'
        )
    python_data = np.asarray(python_eeg['data'], dtype=np.float64)
    matlab_data = np.asarray(saved['data'], dtype=np.float64)
    np.testing.assert_allclose(python_data, matlab_data, rtol=0, atol=1e-7, equal_nan=True)


@pytest.fixture
def eventconsistency_matlab(eeglab_matlab_engine):
    options_path, original = eeg_options_snapshot(eeglab_matlab_engine)
    eeglab_matlab_engine.eval("pop_editoptions('option_single', 0);", nargout=0)
    try:
        yield eeglab_matlab_engine
    finally:
        restore_eeg_options(options_path, original)
        eeglab_matlab_engine.eval('clear functions', nargout=0)


def _matlab_event_rows(saved, *, with_epoch=False):
    event_types = _matlab_strings(saved['event_type'])
    latency = np.asarray(saved['event_latency'], dtype=float).ravel()
    duration = np.asarray(saved['event_duration'], dtype=float).ravel()
    if not with_epoch:
        return [
            (event_types[index], float(latency[index]), float(duration[index])) for index in range(len(event_types))
        ]
    epoch = np.asarray(saved['event_epoch'], dtype=float).ravel()
    return [(event_types[index], float(latency[index]), float(epoch[index])) for index in range(len(event_types))]


def _require_empty_saved_epoch(eeg, name):
    if np.size(eeg.get('epoch')) != 0:
        raise AssertionError(
            f'{name} must save an empty epoch array; a cell of empty structs makes EEGLAB open errordlg2'
        )


def _save_and_compare(engine, eeg, tmp_path, name):
    set_path = tmp_path / f'{name}.set'
    _require_empty_saved_epoch(eeg, name)
    pop_saveset(eeg, str(set_path))
    labels = _eeg_channel_labels(pop_loadset(str(set_path)))
    if not labels:
        raise AssertionError(f'{name} has no EEG-type channels')
    python_eeg = pop_select(pop_loadset(str(set_path)), channel=labels)
    saved = _matlab_pop_select(engine, set_path, labels, tmp_path / f'{name}.mat')
    _assert_matlab_match(python_eeg, saved)


SYNTHETIC_CASES = {
    'leading_boundary': lambda: _dataset(
        [
            {'type': 'boundary', 'latency': 0.5, 'duration': 0},
            {'type': 'stim', 'latency': 10, 'duration': 1},
        ]
    ),
    'duplicate_boundaries': lambda: _dataset(
        [
            {'type': 'boundary', 'latency': 20, 'duration': 4},
            {'type': 'stim', 'latency': 40, 'duration': 0},
            {'type': 'boundary', 'latency': 20, 'duration': 1},
            {'type': 'boundary', 'latency': 20, 'duration': 2},
            {'type': 'stim', 'latency': 10, 'duration': 0},
        ]
    ),
    'invalid_epochs': lambda: _dataset(
        [
            {'type': 'late', 'latency': 30, 'epoch': 4, 'duration': 0},
            {'type': 'second', 'latency': 5, 'epoch': np.int64(2), 'duration': 0},
            {'type': 'bad', 'latency': 12, 'epoch': 0, 'duration': 0},
            {'type': 'first', 'latency': 30, 'epoch': 1, 'duration': 0},
        ],
        trials=3,
        pnts=20,
    ),
    'split_boundary_run': lambda: _dataset(
        [
            {'type': 'boundary', 'latency': 20, 'duration': 4},
            {'type': 'stim', 'latency': 10, 'duration': 0},
            {'type': 'boundary', 'latency': 20, 'duration': 1},
            {'type': 'boundary', 'latency': 30, 'duration': 8},
            {'type': 'boundary', 'latency': 20, 'duration': 2},
        ]
    ),
}


_EPOCH_SORT_CASES = {
    'crossed_epochs': (
        [
            {'type': 'A', 'latency': 75, 'epoch': 2, 'duration': 0},
            {'type': 'B', 'latency': 15, 'epoch': 1, 'duration': 0},
            {'type': 'C', 'latency': 5, 'epoch': 1, 'duration': 0},
            {'type': 'D', 'latency': 145, 'epoch': 3, 'duration': 0},
        ],
        [('C', 5.0, 1.0), ('B', 15.0, 1.0), ('A', 75.0, 2.0), ('D', 145.0, 3.0)],
        [('B', 15.0, 1.0), ('A', 75.0, 2.0), ('C', 5.0, 1.0), ('D', 145.0, 3.0)],
    ),
    'same_epoch_pair': (
        [
            {'type': 'X', 'latency': 85, 'epoch': 2, 'duration': 0},
            {'type': 'Y', 'latency': 72, 'epoch': 2, 'duration': 0},
            {'type': 'Z', 'latency': 8, 'epoch': 1, 'duration': 0},
        ],
        [('Z', 8.0, 1.0), ('Y', 72.0, 2.0), ('X', 85.0, 2.0)],
        [('X', 85.0, 2.0), ('Z', 8.0, 1.0), ('Y', 72.0, 2.0)],
    ),
}


@pytest.mark.parametrize('case_name', list(SYNTHETIC_CASES))
def test_pop_select_synthetic_events_match_matlab(case_name, eventconsistency_matlab, tmp_path):
    _save_and_compare(eventconsistency_matlab, SYNTHETIC_CASES[case_name](), tmp_path, case_name)


@pytest.mark.parametrize('case_name', list(_EPOCH_SORT_CASES))
def test_epoched_sort_differs_from_eeglab(case_name, eventconsistency_matlab, tmp_path):
    events, python_order, matlab_order = _EPOCH_SORT_CASES[case_name]
    eeg = _dataset(events, trials=3, pnts=50)
    _require_empty_saved_epoch(eeg, case_name)
    set_path = tmp_path / f'{case_name}.set'
    pop_saveset(eeg, str(set_path))
    labels = _eeg_channel_labels(pop_loadset(str(set_path)))
    python_rows = _epoch_rows(pop_select(pop_loadset(str(set_path)), channel=labels))
    saved = _matlab_pop_select(eventconsistency_matlab, set_path, labels, tmp_path / f'{case_name}.mat')
    assert python_rows == python_order
    assert _matlab_event_rows(saved, with_epoch=True) == matlab_order


def test_numeric_boundary_merge_differs_from_eeglab(eventconsistency_matlab, tmp_path):
    eeg = _dataset(
        [
            {'type': -99, 'latency': 20, 'duration': 1},
            {'type': -99, 'latency': 20, 'duration': 2},
        ]
    )
    previous = EEG_OPTIONS['option_boundary99']
    EEG_OPTIONS['option_boundary99'] = 1
    try:
        eventconsistency_matlab.eval("pop_editoptions('option_boundary99', 1);", nargout=0)
        _require_empty_saved_epoch(eeg, 'numeric_boundary')
        set_path = tmp_path / 'numeric_boundary.set'
        pop_saveset(eeg, str(set_path))
        labels = _eeg_channel_labels(pop_loadset(str(set_path)))
        python_eeg = pop_select(pop_loadset(str(set_path)), channel=labels)
        saved = _matlab_pop_select(eventconsistency_matlab, set_path, labels, tmp_path / 'numeric_boundary.mat')
    finally:
        EEG_OPTIONS['option_boundary99'] = previous
        eventconsistency_matlab.eval(f"pop_editoptions('option_boundary99', {int(previous)});", nargout=0)

    def _numeric_rows(rows):
        return [(float(event_type), latency, duration) for event_type, latency, duration in rows]

    assert _numeric_rows(_event_rows(python_eeg)) == [(-99.0, 20.0, 3.0)]
    assert _numeric_rows(_matlab_event_rows(saved)) == [(-99.0, 20.0, 1.0), (-99.0, 20.0, 2.0)]


@pytest.mark.parametrize('run', [1, 2])
def test_ds003061_sub001_pop_select_matches_matlab(run, eventconsistency_matlab, eeglab_suite_root, tmp_path):
    directory = Path(eeglab_suite_root) / 'ds003061' / 'sub-001' / 'eeg'
    matches = sorted(directory.glob(f'*_run-{run}_eeg.set'))
    if len(matches) != 1:
        pytest.fail(f'expected one sub-001 run-{run} .set under {directory}, found {[str(path) for path in matches]}')
    imported = pop_load_frombids(
        str(matches[0]),
        bidsmetadata=True,
        bidschanloc=True,
        bidsevent=True,
        eventtype='value',
        verbose=False,
    )
    labels = _eeg_channel_labels(imported)
    if not labels:
        pytest.fail(f'run-{run} import has no EEG-type channels')
    set_path = tmp_path / matches[0].name
    pop_saveset(imported, str(set_path))
    python_eeg = pop_select(pop_loadset(str(set_path)), channel=labels)
    saved = _matlab_pop_select(eventconsistency_matlab, set_path, labels, tmp_path / f'run-{run}.mat')
    _assert_matlab_match(python_eeg, saved)


if __name__ == '__main__':
    unittest.main()
