"""
Test suite for pop_reref.py with MATLAB parity validation.

This module tests the pop_reref function which re-references EEG data
to average reference or to explicit common-reference channels.
"""

import os
import unittest
import sys
import tempfile
import numpy as np
import scipy.io

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.utils.testing import DebuggableTestCase
import importlib
eeg_checkset_module = importlib.import_module('eegprep.functions.adminfunc.eeg_checkset')


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestPopReref(DebuggableTestCase):
    """Test cases for pop_reref function."""

    def setUp(self):
        """Set up test fixtures."""
        # Set up MATLAB compatibility for parity tests
        try:
            self.eeglab = get_eeglab()
            self.matlab_available = True
        except Exception:
            self.matlab_available = False

    def create_test_eeg(self, nbchan=32, pnts=1000, trials=1, srate=256):
        """Create a test EEG structure."""
        np.random.seed(42)  # For reproducible tests

        # Create channel locations
        chanlocs = []
        for i in range(nbchan):
            chanlocs.append({
                'labels': f'Ch{i+1}',
                'X': np.cos(2 * np.pi * i / nbchan),
                'Y': np.sin(2 * np.pi * i / nbchan),
                'Z': 0.0,
                'theta': 2 * np.pi * i / nbchan,
                'radius': 0.5,
                'ref': 'common'  # Initial reference
            })

        # Create ICA components equal to number of channels
        icaweights = np.random.randn(nbchan, nbchan).astype(np.float64)
        icawinv = np.linalg.pinv(icaweights).astype(np.float64)
        icasphere = np.eye(nbchan).astype(np.float64)

        return {
            'data': np.random.randn(nbchan, pnts, trials).astype(np.float64),
            'nbchan': nbchan,
            'pnts': pnts,
            'trials': trials,
            'srate': srate,
            'chanlocs': chanlocs,
            'icaweights': icaweights,
            'icawinv': icawinv,
            'icasphere': icasphere,
            'icachansind': list(range(nbchan)),  # All channels used for ICA
            'ref': 'common'
        }

    def create_simple_eeg(self, nbchan=4, pnts=40):
        """Create a small EEG structure with deterministic channel locations."""
        coords = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.5, 0.5, 0.7071),
        ]
        chanlocs = []
        for idx in range(nbchan):
            x, y, z = coords[idx]
            chanlocs.append({
                'labels': f'Ch{idx + 1}',
                'X': x,
                'Y': y,
                'Z': z,
                'theta': 0.0,
                'radius': 0.5,
                'type': 'EEG',
                'ref': 'common',
            })
        data = np.arange(nbchan * pnts, dtype=np.float64).reshape(nbchan, pnts) / 10.0
        return {
            'data': data,
            'nbchan': nbchan,
            'pnts': pnts,
            'trials': 1,
            'srate': 100,
            'xmin': 0,
            'xmax': (pnts - 1) / 100,
            'chanlocs': chanlocs,
            'urchanlocs': chanlocs.copy(),
            'chaninfo': {},
            'icaweights': np.array([]),
            'icawinv': np.array([]),
            'icasphere': np.array([]),
            'icaact': np.array([]),
            'icachansind': [],
            'ref': 'common',
        }

    @staticmethod
    def _matlab_string(value):
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def _matlab_cellstr(values):
        return "{" + ", ".join(TestPopReref._matlab_string(value) for value in values) + "}"

    @staticmethod
    def _matlab_numvec(values):
        return "[" + " ".join(f"{float(value):.17g}" for value in values) + "]"

    def _matlab_locs_code(self, variable, locs):
        labels = [loc.get('labels', '') for loc in locs]
        types = [loc.get('type', '') for loc in locs]
        refs = [loc.get('ref', '') for loc in locs]
        xs = [loc.get('X', 0.0) for loc in locs]
        ys = [loc.get('Y', 0.0) for loc in locs]
        zs = [loc.get('Z', 0.0) for loc in locs]
        theta = [loc.get('theta', 0.0) for loc in locs]
        radius = [loc.get('radius', 0.5) for loc in locs]
        return f"""
        {variable} = struct('labels', {{}}, 'X', {{}}, 'Y', {{}}, 'Z', {{}}, ...
            'theta', {{}}, 'radius', {{}}, 'type', {{}}, 'ref', {{}});
        tmp_labels = {self._matlab_cellstr(labels)};
        tmp_types = {self._matlab_cellstr(types)};
        tmp_refs = {self._matlab_cellstr(refs)};
        tmp_x = {self._matlab_numvec(xs)};
        tmp_y = {self._matlab_numvec(ys)};
        tmp_z = {self._matlab_numvec(zs)};
        tmp_theta = {self._matlab_numvec(theta)};
        tmp_radius = {self._matlab_numvec(radius)};
        for k = 1:numel(tmp_labels)
            {variable}(k).labels = tmp_labels{{k}};
            {variable}(k).X = tmp_x(k);
            {variable}(k).Y = tmp_y(k);
            {variable}(k).Z = tmp_z(k);
            {variable}(k).theta = tmp_theta(k);
            {variable}(k).radius = tmp_radius(k);
            {variable}(k).type = tmp_types{{k}};
            {variable}(k).ref = tmp_refs{{k}};
        end
        clear tmp_labels tmp_types tmp_refs tmp_x tmp_y tmp_z tmp_theta tmp_radius k;
        """

    def _matlab_pop_reref(self, EEG, ref_expr, option_expr=""):
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        temp_input = tempfile.mktemp(suffix='.mat')
        temp_output = tempfile.mktemp(suffix='.mat')
        icachansind = np.asarray(EEG.get('icachansind', []), dtype=np.float64)
        if icachansind.size:
            icachansind = icachansind + 1
        scipy.io.savemat(temp_input, {
            'data_in': EEG['data'],
            'icaweights_in': EEG.get('icaweights', np.array([])),
            'icawinv_in': EEG.get('icawinv', np.array([])),
            'icasphere_in': EEG.get('icasphere', np.array([])),
            'icaact_in': EEG.get('icaact', np.array([])),
            'icachansind_in': icachansind,
        })

        chanlocs_code = self._matlab_locs_code('EEG.chanlocs', EEG.get('chanlocs', []))
        urchanlocs_code = self._matlab_locs_code('EEG.urchanlocs', EEG.get('urchanlocs', []))
        removed_code = ""
        removedchans = EEG.get('chaninfo', {}).get('removedchans', [])
        if removedchans:
            removed_code = self._matlab_locs_code('EEG.chaninfo.removedchans', removedchans)
        nodatchans_code = ""
        nodatchans = EEG.get('chaninfo', {}).get('nodatchans', [])
        if nodatchans:
            nodatchans_code = self._matlab_locs_code('EEG.chaninfo.nodatchans', nodatchans)

        matlab_code = f"""
        load({self._matlab_string(temp_input)});
        EEG = eeg_emptyset;
        EEG.setname = 'pop_reref parity';
        EEG.data = data_in;
        EEG.nbchan = size(data_in, 1);
        EEG.pnts = size(data_in, 2);
        EEG.trials = 1;
        EEG.srate = {float(EEG.get('srate', 100)):.17g};
        EEG.xmin = {float(EEG.get('xmin', 0)):.17g};
        EEG.xmax = {float(EEG.get('xmax', 0)):.17g};
        EEG.ref = 'common';
        {chanlocs_code}
        {urchanlocs_code}
        EEG.icaweights = icaweights_in;
        EEG.icawinv = icawinv_in;
        EEG.icasphere = icasphere_in;
        EEG.icaact = icaact_in;
        EEG.icachansind = icachansind_in;
        {removed_code}
        {nodatchans_code}
        [EEG, com] = pop_reref(EEG, {ref_expr}{option_expr});
        data_ml = EEG.data;
        nbchan_ml = EEG.nbchan;
        labels_ml = {{EEG.chanlocs.labels}};
        if isempty(EEG.icawinv), icawinv_ml = []; else, icawinv_ml = EEG.icawinv; end
        if isempty(EEG.icaweights), icaweights_ml = []; else, icaweights_ml = EEG.icaweights; end
        if isempty(EEG.icasphere), icasphere_ml = []; else, icasphere_ml = EEG.icasphere; end
        if isempty(EEG.icaact), icaact_ml = []; else, icaact_ml = EEG.icaact; end
        if isempty(EEG.icachansind), icachansind_ml = []; else, icachansind_ml = EEG.icachansind; end
        save({self._matlab_string(temp_output)}, 'data_ml', 'nbchan_ml', 'labels_ml', ...
            'icawinv_ml', 'icaweights_ml', 'icasphere_ml', 'icaact_ml', 'icachansind_ml');
        """
        self.eeglab.engine.eval(matlab_code, nargout=0)
        out = scipy.io.loadmat(temp_output)
        os.remove(temp_input)
        os.remove(temp_output)
        return out

    def test_basic_average_reference_none(self):
        """Test basic average reference with ref=None."""
        EEG = self.create_test_eeg(nbchan=32, pnts=512)
        original_data = EEG['data'].copy()

        result = pop_reref(EEG, ref=None)

        # Check that the function returns a copy (not the same object)
        self.assertIsNot(result, EEG)

        # Check that reference is set to 'average'
        self.assertEqual(result['ref'], 'average')

        # Check that all channel references are updated
        for chan in result['chanlocs']:
            self.assertEqual(chan['ref'], 'average')

        # Check that data is modified (average subtracted)
        self.assertFalse(np.array_equal(original_data, result['data']))

        # Check that the mean across channels is approximately zero
        mean_across_channels = np.mean(result['data'], axis=0)
        np.testing.assert_allclose(mean_across_channels, 0, atol=1e-6)

    def test_basic_average_reference_empty_list(self):
        """Test basic average reference with ref=[]."""
        EEG = self.create_test_eeg(nbchan=16, pnts=256)
        original_data = EEG['data'].copy()

        result = pop_reref(EEG, ref=[])

        # Should behave the same as ref=None
        # Function returns a copy, not the same object
        self.assertIsNot(result, EEG)
        self.assertEqual(result['ref'], 'average')

        # Check that data is modified (average subtracted)
        self.assertFalse(np.array_equal(original_data, result['data']))

        # Check that the mean across channels is approximately zero
        mean_across_channels = np.mean(result['data'], axis=0)
        np.testing.assert_allclose(mean_across_channels, 0, atol=1e-6)

    def test_ica_matrices_updated(self):
        """Test that ICA matrices are properly updated."""
        EEG = self.create_test_eeg(nbchan=16, pnts=256)
        original_icawinv = EEG['icawinv'].copy()
        original_icaweights = EEG['icaweights'].copy()

        result = pop_reref(EEG, ref=None)

        # Check that icawinv is modified (average subtracted)
        self.assertFalse(np.array_equal(original_icawinv, result['icawinv']))

        # Check that icaweights is recomputed
        self.assertFalse(np.array_equal(original_icaweights, result['icaweights']))

        # Check that icasphere is set to identity
        np.testing.assert_array_equal(result['icasphere'], np.eye(EEG['nbchan']))

        # Check that icaweights is the pseudoinverse of icawinv
        computed_weights = np.linalg.pinv(result['icawinv'])
        np.testing.assert_allclose(result['icaweights'], computed_weights, rtol=1e-5)

    def test_icawinv_average_subtraction(self):
        """Test that icawinv has average subtracted correctly."""
        # Disable RMS scaling for this test to check mean subtraction only
        original_option = eeg_checkset_module.option_scaleicarms
        eeg_checkset_module.option_scaleicarms = False
        try:
            EEG = self.create_test_eeg(nbchan=8, pnts=128)
            original_icawinv = EEG['icawinv'].copy()

            result = pop_reref(EEG, ref=None)

            # Check that the mean across channels (axis=0) was subtracted
            expected_icawinv = original_icawinv - np.mean(original_icawinv, axis=0)
            np.testing.assert_allclose(result['icawinv'], expected_icawinv, rtol=1e-6)

            # Check that the mean across channels is approximately zero
            mean_across_channels = np.mean(result['icawinv'], axis=0)
            np.testing.assert_allclose(mean_across_channels, 0, atol=1e-6)
        finally:
            eeg_checkset_module.option_scaleicarms = original_option

    def test_channel_reference_update(self):
        """Test that all channel references are updated to 'average'."""
        EEG = self.create_test_eeg(nbchan=10, pnts=100)

        # Set different initial references
        for i, chan in enumerate(EEG['chanlocs']):
            chan['ref'] = f'ref_{i}'

        result = pop_reref(EEG, ref=None)

        # All channels should now have 'average' reference
        for chan in result['chanlocs']:
            self.assertEqual(chan['ref'], 'average')

    def test_explicit_single_reference_removes_ref_by_default(self):
        """Test common reference to a channel index."""
        EEG = self.create_test_eeg(nbchan=4, pnts=20)
        original_data = EEG['data'].copy()

        result = pop_reref(EEG, ref=1)

        expected = original_data - original_data[1:2, :, :]
        expected = np.delete(expected, 1, axis=0)
        if expected.ndim == 3 and expected.shape[2] == 1:
            expected = np.squeeze(expected, axis=2)
        np.testing.assert_allclose(result['data'], expected, rtol=1e-6)
        self.assertEqual(result['nbchan'], 3)
        self.assertEqual([chan['labels'] for chan in result['chanlocs']], ['Ch1', 'Ch3', 'Ch4'])
        self.assertEqual(result['ref'], 'common')
        self.assertEqual(result['chaninfo']['removedchans'][0]['labels'], 'Ch2')

    def test_explicit_reference_label_keepref(self):
        """Test common reference to a channel label while keeping the reference row."""
        EEG = self.create_test_eeg(nbchan=4, pnts=20)
        original_data = EEG['data'].copy()

        result = pop_reref(EEG, ref='Ch2', keepref='on')

        expected = original_data - original_data[1:2, :, :]
        if expected.ndim == 3 and expected.shape[2] == 1:
            expected = np.squeeze(expected, axis=2)
        np.testing.assert_allclose(result['data'], expected, rtol=1e-6)
        self.assertEqual(result['nbchan'], 4)
        np.testing.assert_allclose(result['data'][1], 0, atol=1e-6)
        for chan in result['chanlocs']:
            self.assertEqual(chan['ref'], 'Ch2')

    def test_average_reference_exclude_leaves_excluded_channel_unchanged(self):
        """Test average reference with excluded channels."""
        EEG = self.create_test_eeg(nbchan=4, pnts=20)
        original_data = EEG['data'].copy()

        result = pop_reref(EEG, ref=[], exclude=[3])

        mean_included = original_data[:3].mean(axis=0)
        expected = original_data.copy()
        expected[:3] = expected[:3] - mean_included
        if expected.ndim == 3 and expected.shape[2] == 1:
            expected = np.squeeze(expected, axis=2)
        np.testing.assert_allclose(result['data'], expected, rtol=1e-6)
        np.testing.assert_allclose(result['data'][3], np.squeeze(original_data[3], axis=-1), rtol=1e-6)
        self.assertNotEqual(result['chanlocs'][3].get('ref'), 'average')

    def test_interpchan_infers_removed_channels_and_removes_them_after_reref(self):
        """Test EEGLAB-style interpolate-reference-remove workflow."""
        EEG = self.create_simple_eeg(nbchan=3, pnts=20)
        removed = {
            'labels': 'Ch4',
            'X': 0.0,
            'Y': -1.0,
            'Z': 0.0,
            'theta': 90.0,
            'radius': 0.5,
            'type': 'EEG',
            'ref': 'common',
        }
        EEG['urchanlocs'] = EEG['chanlocs'] + [removed]
        EEG['chaninfo'] = {'removedchans': [removed]}

        result = pop_reref(EEG, ref=[], interpchan=[])

        self.assertEqual(result['nbchan'], 3)
        self.assertEqual([chan['labels'] for chan in result['chanlocs']], ['Ch1', 'Ch2', 'Ch3'])
        self.assertNotIn('Ch4', [chan['labels'] for chan in result['chanlocs']])

    def test_refloc_adds_old_reference_channel_to_data(self):
        """Test adding a current reference channel back to the data."""
        EEG = self.create_simple_eeg(nbchan=2, pnts=20)
        old_ref = {
            'labels': 'M1',
            'X': 0.0,
            'Y': -1.0,
            'Z': 0.0,
            'theta': -90.0,
            'radius': 0.5,
            'type': 'REF',
            'ref': 'common',
        }
        EEG['chaninfo'] = {
            'nodatchans': [old_ref],
            'removedchans': [old_ref],
        }

        result = pop_reref(EEG, ref=[], refloc='M1')

        self.assertEqual(result['nbchan'], 3)
        self.assertEqual(result['chanlocs'][-1]['labels'], 'M1')
        np.testing.assert_allclose(result['data'].mean(axis=0), 0, atol=1e-6)

    def test_refloc_requires_removed_reference_information_like_eeglab(self):
        """Test EEGLAB error path when refloc is provided without removedchans."""
        EEG = self.create_simple_eeg(nbchan=2, pnts=20)
        EEG['chaninfo'] = {
            'nodatchans': [{
                'labels': 'M1',
                'X': 0.0,
                'Y': -1.0,
                'Z': 0.0,
                'theta': -90.0,
                'radius': 0.5,
                'type': 'REF',
                'ref': 'common',
            }]
        }

        with self.assertRaisesRegex(ValueError, "Missing reference channel information"):
            pop_reref(EEG, ref=[], refloc='M1')

    def test_refica_remove_and_off_modes(self):
        """Test refica options that intentionally do not re-reference ICA maps."""
        original_option = eeg_checkset_module.option_scaleicarms
        eeg_checkset_module.option_scaleicarms = False
        try:
            EEG = self.create_test_eeg(nbchan=4, pnts=20)
            original_icawinv = EEG['icawinv'].copy()
            original_icaweights = EEG['icaweights'].copy()
            EEG['icaact'] = np.ones((4, EEG['pnts']))

            removed = pop_reref(EEG, ref=[], refica='remove')
            self.assertEqual(removed['icawinv'].size, 0)
            self.assertEqual(removed['icaweights'].size, 0)
            self.assertEqual(removed['icasphere'].size, 0)

            off = pop_reref(EEG, ref=[], refica='off')
            np.testing.assert_allclose(off['icawinv'], original_icawinv)
            np.testing.assert_allclose(off['icaweights'], original_icaweights)
            self.assertIsInstance(off['icachansind'], np.ndarray)
        finally:
            eeg_checkset_module.option_scaleicarms = original_option

    def test_refica_clears_when_ica_channel_is_excluded(self):
        """Test EEGLAB edge case where excluded ICA channels invalidate ICA."""
        EEG = self.create_test_eeg(nbchan=4, pnts=20)

        result = pop_reref(EEG, ref=[], exclude=[0])

        self.assertEqual(result['icawinv'].size, 0)
        self.assertEqual(result['icaweights'].size, 0)

    def test_refica_backwardcomp_rereferences_ica_maps(self):
        """Test EEGLAB backwardcomp path still updates ICA maps."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)
        EEG['icawinv'] = np.array([
            [1.0, 0.2, 0.1, 0.0],
            [0.1, 1.0, 0.2, 0.1],
            [0.0, 0.1, 1.0, 0.2],
            [0.2, 0.0, 0.1, 1.0],
        ])
        EEG['icaweights'] = np.linalg.pinv(EEG['icawinv'])
        EEG['icasphere'] = np.eye(4)
        EEG['icaact'] = np.ones((4, EEG['pnts']))
        EEG['icachansind'] = [0, 1, 2, 3]

        result = pop_reref(EEG, ref=[], refica='backwardcomp')

        np.testing.assert_allclose(result['icawinv'].mean(axis=0), 0, atol=1e-8)
        self.assertGreater(result['icaact'].size, 0)
        self.assertEqual(result['icasphere'].shape, (4, 4))

    def test_error_icachansind_mismatch(self):
        """Test behavior when icachansind length doesn't match nbchan."""
        EEG = self.create_test_eeg(nbchan=16, pnts=100)

        # Make icachansind have different length (e.g., after channel removal)
        EEG['icachansind'] = list(range(8))  # Only 8 channels instead of 16

        # The function should clear ICA fields instead of raising an error
        result = pop_reref(EEG, ref=None)

        # Check that ICA fields were cleared
        self.assertEqual(result['icawinv'].size, 0)
        self.assertEqual(result['icaweights'].size, 0)
        self.assertEqual(result['icasphere'].size, 0)

    def test_data_mean_subtraction(self):
        """Test that data has mean subtracted correctly."""
        # Disable RMS scaling for this test to check mean subtraction only
        original_option = eeg_checkset_module.option_scaleicarms
        eeg_checkset_module.option_scaleicarms = False
        try:
            EEG = self.create_test_eeg(nbchan=4, pnts=100)
            original_data = EEG['data'].copy()

            result = pop_reref(EEG, ref=None)

            # Check that the mean across channels (axis=0) was subtracted
            expected_data = original_data - np.mean(original_data, axis=0)
            # eeg_checkset squeezes 3D data with 1 trial to 2D
            if expected_data.ndim == 3 and expected_data.shape[2] == 1:
                expected_data = np.squeeze(expected_data, axis=2)
            np.testing.assert_allclose(result['data'], expected_data, rtol=1e-6)
        finally:
            eeg_checkset_module.option_scaleicarms = original_option

    def test_single_channel(self):
        """Test with single channel (edge case)."""
        EEG = self.create_test_eeg(nbchan=1, pnts=100)
        original_data = EEG['data'].copy()

        result = pop_reref(EEG, ref=None)

        # With single channel, subtracting mean should make data zero
        np.testing.assert_allclose(result['data'], 0, atol=1e-6)

        # Check other fields are updated correctly
        self.assertEqual(result['ref'], 'average')
        self.assertEqual(result['chanlocs'][0]['ref'], 'average')

    def test_multiple_trials(self):
        """Test with multiple trials."""
        EEG = self.create_test_eeg(nbchan=8, pnts=100, trials=5)
        original_data = EEG['data'].copy()

        result = pop_reref(EEG, ref=None)

        # Check that mean is subtracted for each time point and trial
        for trial in range(EEG['trials']):
            for time in range(EEG['pnts']):
                original_mean = np.mean(original_data[:, time, trial])
                new_mean = np.mean(result['data'][:, time, trial])
                self.assertAlmostEqual(new_mean, 0, places=6)

    def test_preserves_data_shape(self):
        """Test that data shape is preserved."""
        EEG = self.create_test_eeg(nbchan=16, pnts=256, trials=3)
        original_shape = EEG['data'].shape

        result = pop_reref(EEG, ref=None)

        self.assertEqual(result['data'].shape, original_shape)

    def test_preserves_other_fields(self):
        """Test that other EEG fields are preserved."""
        EEG = self.create_test_eeg(nbchan=8, pnts=100)
        original_nbchan = EEG['nbchan']
        original_pnts = EEG['pnts']
        original_srate = EEG['srate']
        original_trials = EEG['trials']

        result = pop_reref(EEG, ref=None)

        # These fields should remain unchanged
        self.assertEqual(result['nbchan'], original_nbchan)
        self.assertEqual(result['pnts'], original_pnts)
        self.assertEqual(result['srate'], original_srate)
        self.assertEqual(result['trials'], original_trials)

    def test_deterministic_output(self):
        """Test that function produces deterministic output for same input."""
        EEG = self.create_test_eeg(nbchan=8, pnts=100)

        # Make copies to avoid modification effects
        EEG1 = {key: value.copy() if isinstance(value, np.ndarray) else
                ([item.copy() if isinstance(item, dict) else item for item in value]
                 if isinstance(value, list) else value)
                for key, value in EEG.items()}
        EEG2 = {key: value.copy() if isinstance(value, np.ndarray) else
                ([item.copy() if isinstance(item, dict) else item for item in value]
                 if isinstance(value, list) else value)
                for key, value in EEG.items()}

        result1 = pop_reref(EEG1, ref=None)
        result2 = pop_reref(EEG2, ref=None)

        np.testing.assert_array_equal(result1['data'], result2['data'])
        np.testing.assert_array_equal(result1['icaweights'], result2['icaweights'])
        np.testing.assert_array_equal(result1['icawinv'], result2['icawinv'])

    def test_numerical_precision(self):
        """Test numerical precision of computations."""
        EEG = self.create_test_eeg(nbchan=4, pnts=50)

        result = pop_reref(EEG, ref=None)

        # After average referencing, mean should be very close to zero
        mean_data = np.mean(result['data'], axis=0)
        self.assertTrue(np.all(np.abs(mean_data) < 1e-6))

        mean_icawinv = np.mean(result['icawinv'], axis=0)
        self.assertTrue(np.all(np.abs(mean_icawinv) < 1e-6))

    def test_history_command_formats_label_reference_like_eeglab(self):
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)

        _out, com = pop_reref(EEG, ref='Ch2', keepref='on', return_com=True)

        self.assertEqual(com, "EEG = pop_reref( EEG, {'Ch2'}, 'keepref', 'on');")

    def test_history_command_formats_numeric_channels_as_matlab_indices(self):
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)

        _out, com = pop_reref(EEG, ref=[0], exclude=[3], return_com=True)

        self.assertEqual(com, "EEG = pop_reref( EEG, [1], 'exclude', [4]);")

    def test_history_command_formats_numeric_interpchan_as_matlab_indices(self):
        EEG = self.create_simple_eeg(nbchan=3, pnts=20)
        missing = {'labels': 'Ch4', 'X': 0.0, 'Y': -1.0, 'Z': 0.0, 'theta': 180.0, 'radius': 0.5}
        EEG['urchanlocs'] = EEG['chanlocs'] + [missing]

        _out, com = pop_reref(EEG, ref=[], interpchan=[3], return_com=True)

        self.assertEqual(com, "EEG = pop_reref( EEG, [], 'interpchan', [4]);")

    def test_history_command_formats_refloc_struct_like_eeglab(self):
        EEG = self.create_simple_eeg(nbchan=2, pnts=20)
        old_ref = {
            'labels': 'M1',
            'X': 0.0,
            'Y': -1.0,
            'Z': 0.0,
            'theta': -90.0,
            'radius': 0.5,
            'type': 'REF',
            'ref': 'common',
        }
        EEG['chaninfo'] = {'nodatchans': [old_ref], 'removedchans': [old_ref]}

        _out, com = pop_reref(EEG, ref=[], refloc=old_ref, return_com=True)

        self.assertIn("'refloc', struct(", com)
        self.assertIn("'labels',{'M1'}", com)
        self.assertIn("'theta',-90", com)

    def test_unsupported_legacy_options_raise(self):
        EEG = self.create_simple_eeg(nbchan=2, pnts=20)

        with self.assertRaisesRegex(ValueError, "Unknown pop_reref option"):
            pop_reref(EEG, ref=[], addrefchannel="Cz")
        with self.assertRaisesRegex(ValueError, "Unknown pop_reref option"):
            pop_reref(EEG, ref=[], enforcetype="on")

    def test_multiple_dataset_gui_path_prompts_once_like_eeglab(self):
        class Renderer:
            def __init__(self):
                self.calls = 0

            def run(self, spec, initial_values=None):
                self.calls += 1
                return {"ave": True}

        renderer = Renderer()
        datasets = [self.create_simple_eeg(nbchan=2, pnts=20), self.create_simple_eeg(nbchan=2, pnts=20)]

        outputs, com = pop_reref(datasets, gui=True, renderer=renderer, return_com=True)

        self.assertEqual(renderer.calls, 1)
        self.assertEqual(len(outputs), 2)
        self.assertEqual([out["ref"] for out in outputs], ["average", "average"])
        self.assertEqual(com, "EEG = pop_reref( EEG, []);")

    def test_gui_huber_path_preserves_other_eeglab_gui_options(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "huberef": True,
                    "huberval": "25",
                    "exclude": "1",
                }

        EEG = self.create_simple_eeg(nbchan=2, pnts=20)
        original_excluded = EEG['data'][1].copy()

        out = pop_reref(EEG, gui=True, renderer=Renderer())

        np.testing.assert_allclose(out['data'][1], original_excluded)

    def test_gui_numeric_reference_text_is_zero_based_like_python_api(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "rerefstr": True,
                    "reref": "0",
                    "keepref": True,
                }

        EEG = self.create_simple_eeg(nbchan=3, pnts=20)

        out, com = pop_reref(EEG, gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(com, "EEG = pop_reref( EEG, [1], 'keepref', 'on');")
        np.testing.assert_allclose(out['data'][0], 0, atol=1e-6)

    def test_parity_basic_reref(self):
        """Test parity with MATLAB for basic rereferencing."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Create test data
        EEG = self.create_test_eeg(nbchan=8, pnts=100)

        # Python result
        py_result = pop_reref(EEG.copy(), ref=None)

        # MATLAB result (would need to save EEG structure and call MATLAB)
        # This is a placeholder for the parity test structure
        # ml_result = self.eeglab.pop_reref(EEG, [])

        # For now, just verify Python result is reasonable
        self.assertEqual(py_result['ref'], 'average')
        mean_data = np.mean(py_result['data'], axis=0)
        self.assertTrue(np.all(np.abs(mean_data) < 1e-6))

    def test_parity_data_reref_with_matlab(self):
        """Test parity with MATLAB for data average re-referencing."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        import os
        import tempfile
        import scipy.io
        from eegprep import pop_loadset, pop_saveset

        # Load real test data
        local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')
        test_file = os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')
        EEG = pop_loadset(test_file)

        # Python result
        py_result = pop_reref(EEG.copy(), ref=None)

        # Save to file for MATLAB
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(EEG, temp_file)

        # MATLAB result
        matlab_code = f'''
        EEG = pop_loadset('{temp_file}');
        [~, EEG] = evalc('pop_reref(EEG, []);');
        data_ml = EEG.data;
        save('{temp_file}.mat', 'data_ml');
        '''
        self.eeglab.engine.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        data_ml = mat_data['data_ml']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare data (should match closely for average reference)
        # Data re-referencing: max abs diff ~1e-4 due to float32/float64 precision
        np.testing.assert_allclose(py_result['data'], data_ml, rtol=1e-3, atol=1e-3,
                                   err_msg="Data re-referencing differs from MATLAB")

    def test_parity_explicit_reference_keepref_with_matlab(self):
        """Test MATLAB parity for explicit common reference with keepref."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)

        py_result = pop_reref(EEG, ref=1, keepref='on')
        ml_result = self._matlab_pop_reref(EEG, "2", ", 'keepref', 'on'")

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-6, atol=1e-6)
        self.assertEqual(py_result['nbchan'], int(ml_result['nbchan_ml'].squeeze()))

    def test_parity_explicit_reference_remove_with_matlab(self):
        """Test MATLAB parity for explicit common reference with default ref removal."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)

        py_result = pop_reref(EEG, ref=1)
        ml_result = self._matlab_pop_reref(EEG, "2")

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-6, atol=1e-6)
        self.assertEqual(py_result['nbchan'], int(ml_result['nbchan_ml'].squeeze()))

    def test_parity_exclude_with_matlab(self):
        """Test MATLAB parity for average reference with excluded channels."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)

        py_result = pop_reref(EEG, ref=[], exclude=[3])
        ml_result = self._matlab_pop_reref(EEG, "[]", ", 'exclude', 4")

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-6, atol=1e-6)

    def test_parity_refloc_with_matlab(self):
        """Test MATLAB parity for adding the old reference channel back."""
        EEG = self.create_simple_eeg(nbchan=2, pnts=20)
        old_ref = {
            'labels': 'M1',
            'X': 0.0,
            'Y': -1.0,
            'Z': 0.0,
            'theta': -90.0,
            'radius': 0.5,
            'type': 'REF',
            'ref': 'common',
        }
        EEG['chaninfo'] = {'nodatchans': [old_ref], 'removedchans': [old_ref]}

        py_result = pop_reref(EEG, ref=[], refloc=old_ref)
        ml_result = self._matlab_pop_reref(
            EEG,
            "[]",
            ", 'refloc', EEG.chaninfo.nodatchans",
        )

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-6, atol=1e-6)
        self.assertEqual(py_result['nbchan'], int(ml_result['nbchan_ml'].squeeze()))

    def test_parity_huber_with_matlab(self):
        """Test MATLAB parity for Huber average reference."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)
        EEG['data'][0, :] += 50.0

        py_result = pop_reref(EEG, ref=[], huber=25)
        ml_result = self._matlab_pop_reref(EEG, "[]", ", 'huber', 25")

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-6, atol=1e-6)

    def test_parity_interpchan_with_matlab(self):
        """Test MATLAB parity for interpchan before average reference."""
        EEG = self.create_simple_eeg(nbchan=3, pnts=20)
        removed = {
            'labels': 'Ch4',
            'X': 0.0,
            'Y': -1.0,
            'Z': 0.0,
            'theta': 90.0,
            'radius': 0.5,
            'type': 'EEG',
            'ref': 'common',
        }
        EEG['urchanlocs'] = EEG['chanlocs'] + [removed]
        EEG['chaninfo'] = {'removedchans': [removed]}

        py_result = pop_reref(EEG, ref=[], interpchan=[])
        ml_result = self._matlab_pop_reref(EEG, "[]", ", 'interpchan', []")

        np.testing.assert_allclose(py_result['data'], ml_result['data_ml'], rtol=1e-3, atol=1e-3)
        self.assertEqual(py_result['nbchan'], int(ml_result['nbchan_ml'].squeeze()))

    def test_parity_refica_average_with_matlab(self):
        """Test MATLAB parity for average reference ICA map update."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)
        EEG['icawinv'] = np.array([
            [1.0, 0.2, 0.1, 0.0],
            [0.1, 1.0, 0.2, 0.1],
            [0.0, 0.1, 1.0, 0.2],
            [0.2, 0.0, 0.1, 1.0],
        ])
        EEG['icaweights'] = np.linalg.pinv(EEG['icawinv'])
        EEG['icasphere'] = np.eye(4)
        EEG['icaact'] = np.ones((4, EEG['pnts']))
        EEG['icachansind'] = [0, 1, 2, 3]

        py_result = pop_reref(EEG, ref=[])
        ml_result = self._matlab_pop_reref(EEG, "[]")

        np.testing.assert_allclose(py_result['icawinv'], ml_result['icawinv_ml'], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(py_result['icaweights'], ml_result['icaweights_ml'], rtol=1e-8, atol=1e-8)

    def test_parity_refica_backwardcomp_with_matlab(self):
        """Test MATLAB parity for backwardcomp ICA handling."""
        EEG = self.create_simple_eeg(nbchan=4, pnts=20)
        EEG['icawinv'] = np.array([
            [1.0, 0.2, 0.1, 0.0],
            [0.1, 1.0, 0.2, 0.1],
            [0.0, 0.1, 1.0, 0.2],
            [0.2, 0.0, 0.1, 1.0],
        ])
        EEG['icaweights'] = np.linalg.pinv(EEG['icawinv'])
        EEG['icasphere'] = np.eye(4)
        EEG['icaact'] = np.ones((4, EEG['pnts']))
        EEG['icachansind'] = [0, 1, 2, 3]

        py_result = pop_reref(EEG, ref=[], refica='backwardcomp')
        ml_result = self._matlab_pop_reref(EEG, "[]", ", 'refica', 'backwardcomp'")

        np.testing.assert_allclose(py_result['icawinv'], ml_result['icawinv_ml'], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(py_result['icaweights'], ml_result['icaweights_ml'], rtol=1e-8, atol=1e-8)
        self.assertEqual(py_result['icaact'].size, ml_result['icaact_ml'].size)


if __name__ == '__main__':
    unittest.main()
