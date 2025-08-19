# test_pop_select.py
import unittest
import numpy as np
import copy
import os

from eegprep.eeglabcompat import get_eeglab
from eegprep.pop_loadset import pop_loadset
from eegprep.pop_select import pop_select

# where the test resources
web_root = 'https://sccntestdatasets.s3.us-east-2.amazonaws.com/'
local_url = os.path.join(os.path.dirname(__file__), '../data/')

def ensure_file(fname: str) -> str: # duplicate of test_clean_rawdata.py
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


# class TestPopSelectParity(unittest.TestCase):

#     def setUp(self):
#         # Load the same dataset in both backends
#         self.EEG_py = pop_loadset(ensure_file('FlankerTest.set'))
#         self.ee_mat = get_eeglab('MAT')       # MATLAB bridge
#         self.EEG_mat = self.ee_mat.pop_loadset(ensure_file('FlankerTest.set'))

#     def test_parity_channel_by_name(self):
#         # Keep first 3 channels by name to avoid 0 vs 1-based index differences
#         labels = _chan_labels(self.EEG_py)
#         self.assertGreaterEqual(len(labels), 3, "Dataset must have at least 3 channels")
#         keep_names = labels[:3]

#         EEG_py_in = copy.deepcopy(self.EEG_py)
#         EEG_py_out = pop_select(EEG_py_in, channel=keep_names)

#         EEG_mat_out = self.ee_mat.pop_select(copy.deepcopy(self.EEG_mat), 'channel', keep_names)

#         # Shapes and metadata
#         self.assertEqual(EEG_py_out['nbchan'], 3)
#         self.assertEqual(EEG_mat_out['nbchan'], 3)
#         self.assertEqual(EEG_py_out['pnts'], EEG_mat_out['pnts'])
#         self.assertEqual(EEG_py_out['trials'], EEG_mat_out['trials'])
#         self.assertTrue(np.allclose(EEG_py_out['xmin'], EEG_mat_out['xmin'], atol=1e-12))
#         self.assertTrue(np.allclose(EEG_py_out['xmax'], EEG_mat_out['xmax'], atol=1e-12))
#         # Data shape
#         self.assertEqual(EEG_py_out['data'].shape, EEG_mat_out['data'].shape)
#         # Compare data numerically within tolerance
#         self.assertTrue(np.allclose(EEG_py_out['data'], EEG_mat_out['data'], atol=1e-7, equal_nan=True))
#         # Labels match
#         self.assertEqual(_chan_labels(EEG_py_out), _chan_labels(EEG_mat_out))

#     def test_parity_trial_subset(self):
#         trials = int(self.EEG_py.get('trials', 1))
#         if trials <= 1:
#             self.skipTest("Dataset is continuous; skipping trial subset parity test")
#         k = min(5, trials)
#         keep_trials = list(range(1, k + 1))  # 1-based

#         EEG_py_out, _ = pop_select(copy.deepcopy(self.EEG_py), trial=keep_trials)
#         EEG_mat_out = self.ee_mat.pop_select(copy.deepcopy(self.EEG_mat), 'trial', keep_trials)

#         self.assertEqual(EEG_py_out['trials'], k)
#         self.assertEqual(EEG_mat_out['trials'], k)
#         self.assertEqual(EEG_py_out['nbchan'], EEG_mat_out['nbchan'])
#         self.assertEqual(EEG_py_out['pnts'], EEG_mat_out['pnts'])
#         self.assertTrue(np.allclose(EEG_py_out['data'], EEG_mat_out['data'], atol=1e-7, equal_nan=True))

#         # If events exist, ensure counts match
#         if EEG_py_out.get('event') is not None and EEG_mat_out.get('event') is not None:
#             self.assertEqual(len(EEG_py_out['event']), len(EEG_mat_out['event']))

#     def test_parity_time_selection(self):
#         # Works for both continuous and epoched
#         xmin = float(self.EEG_py['xmin'])
#         xmax = float(self.EEG_py['xmax'])
#         # choose a conservative window inside bounds
#         tmin = xmin
#         tmax = xmin + min(0.2, max(0.05, xmax - xmin))

#         EEG_py_out, _ = pop_select(copy.deepcopy(self.EEG_py), time=np.array([[tmin, tmax]], dtype=float))
#         EEG_mat_out = self.ee_mat.pop_select(copy.deepcopy(self.EEG_mat), 'time', [tmin, tmax])

#         self.assertTrue(np.allclose(EEG_py_out['xmin'], tmin, atol=1e-12))
#         # EEGLAB inclusive endpoint leads to xmax aligned to sample grid
#         self.assertTrue(abs(EEG_py_out['xmax'] - EEG_mat_out['xmax']) < 1.0 / max(self.EEG_py['srate'], 1))
#         self.assertEqual(EEG_py_out['nbchan'], EEG_mat_out['nbchan'])
#         self.assertEqual(EEG_py_out['trials'], EEG_mat_out['trials'])
#         self.assertEqual(EEG_py_out['pnts'], EEG_mat_out['pnts'])
#         self.assertTrue(np.allclose(EEG_py_out['data'], EEG_mat_out['data'], atol=1e-7, equal_nan=True))

#     def test_parity_rmtime_continuous(self):
#         # Only meaningful for continuous data
#         if int(self.EEG_py.get('trials', 1)) > 1:
#             self.skipTest("Dataset is epoched; skipping continuous rmtime parity test")

#         xmin = float(self.EEG_py['xmin'])
#         xmax = float(self.EEG_py['xmax'])
#         span = xmax - xmin
#         if span <= 0.3:
#             self.skipTest("Not enough duration to remove a middle segment")

#         # Remove a middle slice
#         rm_seg = np.array([[xmin + 0.1 * span, xmin + 0.2 * span]], dtype=float)

#         EEG_py_out, _ = pop_select(copy.deepcopy(self.EEG_py), rmtime=rm_seg)
#         EEG_mat_out = self.ee_mat.pop_select(copy.deepcopy(self.EEG_mat), 'rmtime', rm_seg.tolist())

#         self.assertTrue(EEG_py_out['pnts'] < self.EEG_py['pnts'])
#         self.assertEqual(EEG_py_out['pnts'], EEG_mat_out['pnts'])
#         self.assertEqual(EEG_py_out['nbchan'], EEG_mat_out['nbchan'])
#         self.assertTrue(np.allclose(EEG_py_out['data'], EEG_mat_out['data'], atol=1e-7, equal_nan=True))


class TestPopSelectFunctional(unittest.TestCase):

    def setUp(self):
        self.EEG = pop_loadset(ensure_file('FlankerTest.set'))

    def test_channel_and_trial_selection_consistency(self):
        labels = _chan_labels(self.EEG)
        self.assertGreaterEqual(len(labels), 2, "Dataset must have at least 2 channels")
        keep_names = labels[:2]

        trials = int(self.EEG.get('trials', 1))
        if trials > 1:
            keep_trials = list(range(1, min(3, trials) + 1))
        else:
            keep_trials = [1]

        EEG_out = pop_select(copy.deepcopy(self.EEG), channel=keep_names, trial=keep_trials)

        self.assertEqual(EEG_out['nbchan'], 2)
        self.assertEqual(EEG_out['trials'], len(keep_trials))
        self.assertEqual(EEG_out['data'].shape[0], 2)
        if EEG_out['trials'] > 1:
            self.assertEqual(EEG_out['data'].shape[2], len(keep_trials))

        # Events, if present, must have valid latencies
        if EEG_out.get('event'):
            total_pts = EEG_out['pnts'] * EEG_out['trials']
            for ev in EEG_out['event']:
                if 'latency' in ev:
                    self.assertTrue(1 <= float(ev['latency']) <= total_pts)

    def test_time_then_point_precedence(self):
        # If both point and time are provided, point should take precedence for computing time
        srate = float(self.EEG['srate'])
        xmin = float(self.EEG['xmin'])
        # choose a 10-sample window via points
        pt_start = 1
        pt_end = min(self.EEG['pnts'], 10)
        EEG_out = pop_select(copy.deepcopy(self.EEG), point=[pt_start, pt_end], time=[[xmin, xmin + 1000.0]])  # absurd time to force precedence

        # Expect exactly pt_end - pt_start + 1 samples in output
        expected_pnts = pt_end - pt_start + 1
        self.assertEqual(EEG_out['pnts'], expected_pnts)
        # xmin should match eeg_point2lat mapping of pt_start
        # xmin_out equals old xmin plus (pt_start-1)/srate
        self.assertAlmostEqual(EEG_out['xmin'], xmin + (pt_start - 1) / srate, places=7)

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

    def test_event_epoch_field_removal_on_single_trial(self):
        EEG = copy.deepcopy(self.EEG)
        if int(EEG.get('trials')) > 2:
            EEG_out = pop_select(EEG, trial=[1,2])

            self.assertTrue(EEG_out.get('epoch') == [])


if __name__ == '__main__':
    unittest.main()