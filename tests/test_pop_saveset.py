import os
import tempfile
import unittest

import numpy as np
import scipy.io

from eegprep import pop_loadset, pop_saveset  # Explicitly import pop_resample


# where the test resources
local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')

# where the test resources
web_root = 'https://sccntestdatasets.s3.us-east-2.amazonaws.com/'
local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')


def ensure_file(fname: str) -> str:
    """Download a file if it does not exist and return the local path."""
    full_url = f"{web_root}{fname}"
    local_file = f"{local_url}{fname}"
    if not os.path.exists(local_file):
        from urllib.request import urlretrieve

        urlretrieve(full_url, local_file)
    return local_file


class TestPopSaveset(unittest.TestCase):
    def setUp(self):
        pass

    def test_flanker(self):
        pass
        self.EEG = pop_loadset(ensure_file('FlankerTest.set'))
        pop_saveset(
            self.EEG, os.path.join(local_url, 'eeglab_data_tmp.set')
        )  # see MATLAB code to compare the results at the end of the file

    def test_basic(self):
        self.EEG = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set'))
        pop_saveset(
            self.EEG, os.path.join(local_url, 'eeglab_data_tmp.set')
        )  # see MATLAB code to compare the results at the end of the file

    def test_saveset_does_not_mutate_caller_indices(self):
        EEG = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set'))

        chanlocs = EEG['chanlocs']
        events = EEG['event']
        urchan_before = [int(c['urchan']) for c in chanlocs if 'urchan' in c]
        urevent_before = [int(ev['urevent']) for ev in events if 'urevent' in ev]
        latency_before = [float(ev['latency']) for ev in events if 'latency' in ev]
        self.assertTrue(urchan_before, "Dataset must have urchan values to exercise the regression")
        self.assertTrue(urevent_before, "Dataset must have urevent values to exercise the regression")

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'roundtrip.set')
            pop_saveset(EEG, out)
            # A second save must not double-increment the caller's in-memory indices.
            pop_saveset(EEG, out)

        urchan_after = [int(c['urchan']) for c in EEG['chanlocs'] if 'urchan' in c]
        urevent_after = [int(ev['urevent']) for ev in EEG['event'] if 'urevent' in ev]
        latency_after = [float(ev['latency']) for ev in EEG['event'] if 'latency' in ev]

        self.assertEqual(urchan_after, urchan_before)  # still 0-based in memory
        self.assertEqual(urevent_after, urevent_before)
        np.testing.assert_array_equal(latency_after, latency_before)
        # """Test basic resampling functionality with different engines"""
        # # Apply resampling with different engines
        # EEG_python = pop_resample(self.EEG.copy(), self.new_freq, engine='scipy')
        # pop_saveset(EEG_python, os.path.join(local_url, 'eeglab_data_with_ica_tmp_python.set')) # see MATLAB code to compare the results at the end of the file

        # EEG_matlab = pop_resample(self.EEG.copy(), self.new_freq, engine='matlab')
        # pop_saveset(EEG_matlab, os.path.join(local_url, 'eeglab_data_with_ica_tmp_matlab.set')) # see MATLAB code to compare the results at the end of the file

        # EEG_octave = pop_resample(self.EEG.copy(), self.new_freq, engine='octave')
        # pop_saveset(EEG_octave, os.path.join(local_url, 'eeglab_data_with_ica_tmp_octave.set')) # see MATLAB code to compare the results at the end of the file

        # # Check sampling rates
        # self.assertEqual(EEG_python['srate'], self.new_freq, 'Python resampling failed')
        # print("Sampling rate ok")
        # self.assertEqual(EEG_matlab['srate'], self.new_freq, 'MATLAB resampling failed')
        # print("Sampling rate ok")
        # self.assertEqual(EEG_octave['srate'], self.new_freq, 'Octave resampling failed')
        # print("Sampling rate ok")

        # # Compare data shapes
        # self.assertEqual(EEG_python['data'].shape[1], EEG_matlab['data'].shape[1],
        #                 'Data shapes differ between Python and MATLAB')
        # print("Data shape ok")
        # self.assertEqual(EEG_python['data'].shape[1], EEG_octave['data'].shape[1],
        #                 'Data shapes differ between Python and Octave')
        # print("Data shape ok")
        # # Compare data with tolerance
        # # np.testing.assert_allclose(EEG_python['data'], EEG_matlab['data'],
        # #                          rtol=1e-5, atol=1e-8,
        # #                          err_msg='Python and MATLAB results differ beyond tolerance')
        # np.testing.assert_allclose(EEG_matlab['data'].flatten(), EEG_octave['data'].flatten(),
        #                          rtol=1e-5, atol=1e-8,
        #                          err_msg='Python and MATLAB results differ beyond tolerance')
        # print("Data comparison ok")

        # Compare ICA activations if present
        # if 'icaact' in self.EEG:
        #     np.testing.assert_allclose(EEG_python['icaact'], EEG_matlab['icaact'],
        #                              rtol=1e-5, atol=1e-8,
        #                              err_msg='ICA activations differ between Python and MATLAB')
        #     np.testing.assert_allclose(EEG_python['icaact'], EEG_octave['icaact'],
        #                              rtol=1e-5, atol=1e-8,
        #                              err_msg='ICA activations differ between Python and Octave')

    def test_chanlocs_serialized_through_single_converter(self):
        # The primary EEG.chanlocs struct must be serialized through the same
        # canonical converter as chaninfo.removedchans, so a chanloc field such
        # as ``unit`` is preserved instead of being dropped by a second copy.
        chanlocs = [
            {
                'labels': 'Fz',
                'theta': 0.0,
                'radius': 0.5,
                'X': 1.0,
                'Y': 2.0,
                'Z': 3.0,
                'sph_theta': 0.0,
                'sph_phi': 0.0,
                'sph_radius': 1.0,
                'type': 'EEG',
                'urchan': 0,
                'ref': np.array([]),
                'unit': 'uV',
            },
            {
                'labels': 'Cz',
                'theta': 10.0,
                'radius': 0.6,
                'X': 1.5,
                'Y': 2.5,
                'Z': 3.5,
                'sph_theta': 1.0,
                'sph_phi': 1.0,
                'sph_radius': 1.0,
                'type': 'EEG',
                'urchan': 1,
                'ref': np.array([]),
                'unit': 'uV',
            },
        ]
        EEG = {
            'setname': 't',
            'nbchan': 2,
            'trials': 1,
            'pnts': 4,
            'srate': 100.0,
            'xmin': 0.0,
            'xmax': 0.03,
            'times': np.arange(4) / 100.0,
            'data': np.zeros((2, 4)),
            'chanlocs': chanlocs,
            'event': [],
            'icachansind': np.array([]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'unit.set')
            pop_saveset(EEG, out)
            loaded = scipy.io.loadmat(out, struct_as_record=True)
        self.assertIn('unit', loaded['chanlocs'].dtype.names)

    def test_no_location_channel_coordinates_saved_as_empty(self):
        # No-location channels (e.g. EOG) must keep empty coordinates on save, as
        # EEGLAB does. Writing 0 would place them at the head center when the .set
        # is reloaded (here or in EEGLAB), corrupting scalp maps.
        empty = np.array([])
        chanlocs = [
            {'labels': 'Cz', 'theta': 0.0, 'radius': 0.0, 'X': 0.0, 'Y': 0.0, 'Z': 1.0},
            {'labels': 'EOG', 'theta': empty, 'radius': empty, 'X': empty, 'Y': empty, 'Z': empty},
        ]
        EEG = {
            'setname': 't',
            'nbchan': 2,
            'trials': 1,
            'pnts': 4,
            'srate': 100.0,
            'xmin': 0.0,
            'xmax': 0.03,
            'times': np.arange(4) / 100.0,
            'data': np.zeros((2, 4)),
            'chanlocs': chanlocs,
            'event': [],
            'icachansind': np.array([]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'noloc.set')
            pop_saveset(EEG, out)
            raw = scipy.io.loadmat(out, struct_as_record=False, squeeze_me=True)['chanlocs']
            reloaded = pop_loadset(out)['chanlocs']
        # On disk: no-location coords are empty (not 0); the located channel keeps its value.
        self.assertEqual(np.size(raw[1].theta), 0)
        self.assertEqual(np.size(raw[1].radius), 0)
        self.assertEqual(float(raw[0].radius), 0.0)
        # Round-trip through EEGPrep keeps the no-location coords empty.
        self.assertEqual(np.asarray(reloaded[0]['radius']).size, 1)
        self.assertEqual(np.asarray(reloaded[1]['theta']).size, 0)
        self.assertEqual(np.asarray(reloaded[1]['radius']).size, 0)


if __name__ == '__main__':
    # EEG = pop_loadset(ensure_file('FlankerTest.set'))
    # pop_saveset(EEG, os.path.join(local_url, 'eeglab_data_tmp.set')) # see MATLAB code to compare the results at the end of the file

    unittest.main()

# MATLAB code to compare the results
# EEG_matlab = pop_loadset('eeglab_data_with_ica_tmp_matlab.set');
# EEG_octave = pop_loadset('eeglab_data_with_ica_tmp_octave.set');
# EEG_python = pop_loadset('eeglab_data_with_ica_tmp_python.set');
# eegplot(EEG_matlab.data, 'srate', EEG_matlab.srate, 'data2', EEG_python.data);
# figure; plot(EEG_matlab.data(1:10000), EEG_python.data(1:10000),'.')

# figure; hist(abs(EEG_octave.data(:) - EEG_matlab.data(:)), 100)

# assert(all( abs(EEG_python.data(:) - EEG_matlab.data(:)) <= (1e-8 + 1e-5 * abs(EEG_matlab.data(:))) ), ...
#        'Python and MATLAB results differ beyond tolerance');
