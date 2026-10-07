# test_pop_select.py
import os
import unittest
import numpy as np
import copy

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.functions.popfunc.pop_epoch import pop_epoch
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
        # Load the same dataset in both backends
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

    # TODO: This test has pre-existing issues with boundary adjustment differences


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

        # Select first two trials
        EEG_out = pop_select(EEG, trial=[1, 2])

        self.assertEqual(EEG_out['trials'], 2)
        # Should have 2 events (from epochs 1 and 2)
        self.assertEqual(len(EEG_out['event']), 2)
        # Check epoch numbers are updated
        self.assertEqual(EEG_out['event'][0]['epoch'], 1)
        self.assertEqual(EEG_out['event'][1]['epoch'], 2)

    def test_dipfit_removal_warning(self):
        """Test that dipfit is removed when channels are removed."""
        EEG = copy.deepcopy(self.EEG)
        EEG['dipfit'] = [{'some': 'dipole_info'}]
        EEG['roi'] = [{'some': 'roi_info'}]

        # Remove one channel
        EEG_out = pop_select(EEG, channel=[0, 1, 2])  # Remove channel 3

        # dipfit and roi should be cleared
        self.assertEqual(np.asarray(EEG_out['dipfit']).size, 0)
        self.assertEqual(EEG_out['roi'], {})

    def test_single_trial_epoch_field_removal(self):
        """Test that epoch fields are removed from events when only one trial remains."""
        EEG = copy.deepcopy(self.EEG)

        # Select only first trial
        EEG_out = pop_select(EEG, trial=[1])

        self.assertEqual(EEG_out['trials'], 1)
        # epoch field should be removed from events
        for event in EEG_out['event']:
            self.assertNotIn('epoch', event)
        # epoch list should be empty
        self.assertEqual(np.asarray(EEG_out['epoch']).size, 0)


if __name__ == '__main__':
    unittest.main()
