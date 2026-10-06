import copy
import os
import unittest
import numpy as np

from eegprep.functions.popfunc.eeg_interp import eeg_interp, spheric_spline, computeg
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_select import pop_select
from tests.eeglab_tests import eeglab_test

# Test Case	(Python vs. MATLAB)         Max Absolute	Max Relative	Scenario
# test_parity_multiple_trials	        5.11e-04	1.99e-02 (1.99%)	3 trials, 3 channels, 1 trial
# test_parity_spherical_crd	            3.89e-04	2.76e-03 (0.28%)	SphericalCRD method, 3 channels, 1 trial
# test_parity_custom_params	            3.43e-04	2.08e-03 (0.21%)	Custom parameters, 3 channels, 1 trial
# test_parity_spherical_kang	        2.81e-04	2.98e-03 (0.30%)	SphericalKang method, 3 channels, 1 trial
# test_parity_spherical_basic	        2.50e-04	3.18e-03 (0.32%)	Basic spherical, 3 channels, 1 trial
# test_parity_custom_time_range	        2.49e-04	2.02e-03 (0.20%)	Custom time range, 3 channels


def test_current_suite_checkinterp_preserves_existing_channels_when_restoring_montage():
    eeg = pop_loadset("sample_data/eeglab_data.set")
    eeg["data"] = eeg["data"][:, :1000]
    eeg["pnts"] = 1000
    original_locs = eeg["chanlocs"]
    reduced = pop_select(eeg, nochannel=list(range(0, eeg["nbchan"], 4)))

    restored = eeg_interp(reduced, original_locs)

    restored_labels = [loc["labels"] for loc in restored["chanlocs"]]
    reduced_labels = [loc["labels"] for loc in reduced["chanlocs"]]
    for reduced_index, label in enumerate(reduced_labels):
        restored_index = restored_labels.index(label)
        np.testing.assert_array_equal(restored["data"][restored_index], reduced["data"][reduced_index])


@eeglab_test("unittesting_popfunc/eeg_interp/popfunc_eeg_interp_wrapperTest.m", "test_test_eeg_interp")
def test_reference_interp_original_methods_and_shuffled_montage(eeglab_backend, eeglab_suite_root):
    directory = eeglab_suite_root / "eeglab/sample_data"
    eeg = eeglab_backend("pop_loadset", str(directory / "eeglab_data.set"))
    eeg["chanlocs"] = eeglab_backend(
        "pop_chanedit",
        eeg["chanlocs"],
        "load",
        np.array([[str(directory / "eeglab_chan32.locs"), "filetype", ""]], dtype=object),
        "shrink",
        -0.1,
    )
    eeg["pnts"] = 1000.0
    eeg["data"] = eeg["data"][:, :1000]
    eeg = eeglab_backend("eeg_checkset", eeg)
    for method in ("spherical", "invdist"):
        eeglab_backend("eeg_interp", eeg, np.arange(1.0, 17.0)[None, :], method)
        eeglab_backend("eeg_interp", eeg, np.empty((0, 0)), method)
    locations = eeg["chanlocs"]
    eeg = eeglab_backend(
        "pop_select", eeg, "nochannel", np.arange(1.0, float(np.asarray(eeg["nbchan"]).item()) + 1, 4)[None, :]
    )
    _reference_checkinterp(eeg, eeglab_backend("eeg_interp", eeg, locations))
    shuffled = eeglab_backend("shuffle", locations)
    _reference_checkinterp(eeg, eeglab_backend("eeg_interp", eeg, shuffled))


def _reference_checkinterp(first, second):
    first_labels = first["chanlocs"]["labels"].ravel(order="F").tolist()
    second_labels = second["chanlocs"]["labels"].ravel(order="F").tolist()
    if len(first_labels) > len(second_labels):
        _reference_checkinterp(second, first)
        return
    for index, label in enumerate(first_labels):
        matches = [other for other, candidate in enumerate(second_labels) if candidate == label]
        if matches:
            # checkinterp.m checks only the first sample of matching channels.
            assert np.all(first["data"][index, 0] == second["data"][matches, 0])


class TestEegInterpPlanarGeometry(unittest.TestCase):
    def test_planar_theta_radius_degrees_match_xy_fallback(self):
        n_channels = 6
        n_points = 24
        data = np.arange(n_channels * n_points, dtype=float).reshape(n_channels, n_points)
        chanlocs = []
        for theta in np.linspace(-150, 150, n_channels):
            theta_rad = np.deg2rad(theta)
            radius = 0.45
            x = radius * np.sin(theta_rad)
            y = radius * np.cos(theta_rad)
            chanlocs.append(
                {
                    "labels": f"Ch{len(chanlocs) + 1}",
                    "theta": theta,
                    "radius": radius,
                    "X": x,
                    "Y": y,
                    "Z": 0.75,
                }
            )

        eeg_theta = {
            "data": data.copy(),
            "nbchan": n_channels,
            "pnts": n_points,
            "trials": 1,
            "srate": 100,
            "xmin": 0.0,
            "xmax": (n_points - 1) / 100,
            "chanlocs": chanlocs,
        }
        eeg_xy = copy.deepcopy(eeg_theta)
        for loc in eeg_xy["chanlocs"]:
            loc["theta"] = []
            loc["radius"] = []

        by_theta = eeg_interp(eeg_theta, [0], method="invdist")
        by_xy = eeg_interp(eeg_xy, [0], method="invdist")

        np.testing.assert_allclose(by_theta["data"], by_xy["data"], rtol=1e-12, atol=1e-12)


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestEegInterpParity(unittest.TestCase):
    """Test parity between Python eeg_interp and MATLAB eeg_interp.m"""

    def setUp(self):
        """Set up MATLAB interface and test EEG data"""
        self.eeglab = get_eeglab('MAT')

        # Create a simple EEG structure for parity testing
        n_channels = 32
        n_timepoints = 1000
        n_trials = 1

        # Generate synthetic EEG data
        np.random.seed(42)  # For reproducible tests
        self.test_EEG = {
            'data': np.random.randn(n_channels, n_timepoints, n_trials) * 50,
            'nbchan': n_channels,
            'pnts': n_timepoints,
            'trials': n_trials,
            'srate': 500,
            'xmin': -1.0,
            'xmax': 1.0,
            'times': np.linspace(-1.0, 1.0, n_timepoints),
            'chanlocs': [],
            # Required fields for pop_saveset
            'icaact': np.array([]),
            'icawinv': np.array([]),
            'icasphere': np.array([]),
            'icaweights': np.array([]),
            'icachansind': np.array([]),
            'urchanlocs': [],
            'chaninfo': {},
            'ref': 'common',
            'history': '',
            'saved': 'no',
            'etc': {},
        }

        # Create realistic channel locations on unit sphere
        for i in range(n_channels):
            theta = 2 * np.pi * i / n_channels
            phi = np.pi / 6 + (np.pi / 3) * (i % 8) / 8

            x = np.cos(phi) * np.cos(theta)
            y = np.cos(phi) * np.sin(theta)
            z = np.sin(phi)

            # Add some standard channel names for first few channels
            if i == 0:
                label = 'Fp1'
            elif i == 1:
                label = 'Fp2'
            elif i == 2:
                label = 'F7'
            elif i == 3:
                label = 'F3'
            else:
                label = f'Ch{i + 1}'

            self.test_EEG['chanlocs'].append(
                {
                    'labels': label,
                    'X': x,
                    'Y': y,
                    'Z': z,
                    'theta': np.arctan2(y, x),
                    'radius': np.sqrt(x**2 + y**2),
                    'sph_theta': theta,
                    'sph_phi': phi,
                    'sph_radius': 1.0,  # Unit sphere
                    'type': 'EEG',
                    'urchan': i,  # 0-based original channel index
                    'ref': '',
                }
            )

    def _compare_eeg_results(self, py_result, ml_result):
        """Helper method to compare Python and MATLAB EEG results"""
        # Compare interpolated data (handle shape differences for single trial)
        if py_result['data'].ndim == 3 and ml_result['data'].ndim == 2 and py_result['trials'] == 1:
            # MATLAB returns 2D for single trial, Python returns 3D
            py_data_2d = py_result['data'][:, :, 0]  # Extract single trial
            self.assertEqual(py_data_2d.shape, ml_result['data'].shape)

            # Check if the data is close (allow for numerical differences)
            max_abs_diff = np.max(np.abs(py_data_2d - ml_result['data']))
            max_rel_diff = np.max(np.abs(py_data_2d - ml_result['data']) / (np.abs(ml_result['data']) + 1e-12))

            # Allow for reasonable numerical differences in interpolation
            self.assertLess(max_abs_diff, 5e-2, f"Max absolute difference: {max_abs_diff}")
            self.assertLess(max_rel_diff, 5e-2, f"Max relative difference: {max_rel_diff}")
        else:
            self.assertEqual(py_result['data'].shape, ml_result['data'].shape)
            max_abs_diff = np.max(np.abs(py_result['data'] - ml_result['data']))
            max_rel_diff = np.max(np.abs(py_result['data'] - ml_result['data']) / (np.abs(ml_result['data']) + 1e-12))

            self.assertLess(max_abs_diff, 5e-2, f"Max absolute difference: {max_abs_diff}")
            self.assertLess(max_rel_diff, 5e-2, f"Max relative difference: {max_rel_diff}")

        # Compare structure fields
        self.assertEqual(py_result['nbchan'], ml_result['nbchan'])
        self.assertEqual(py_result['pnts'], ml_result['pnts'])
        self.assertEqual(py_result['trials'], ml_result['trials'])

    def test_parity_spherical_basic(self):
        """Test parity for basic spherical interpolation with channel indices"""
        bad_chans = [0, 1, 2]  # First 3 channels (0-based for Python)
        bad_chans_matlab = [1, 2, 3]  # 1-based for MATLAB

        # Python interpolation
        py_result = eeg_interp(self.test_EEG, bad_chans, method='spherical')

        # MATLAB interpolation
        ml_result = self.eeglab.eeg_interp(self.test_EEG, bad_chans_matlab, 'spherical')

        # Compare results
        self._compare_eeg_results(py_result, ml_result)

    def test_parity_spherical_kang(self):
        """Test parity for sphericalKang method"""
        bad_chans = [5, 10]
        bad_chans_matlab = [6, 11]  # 1-based for MATLAB

        py_result = eeg_interp(self.test_EEG, bad_chans, method='sphericalKang')
        ml_result = self.eeglab.eeg_interp(self.test_EEG, bad_chans_matlab, 'sphericalKang')

        # Compare results using the helper method
        self._compare_eeg_results(py_result, ml_result)

    def test_parity_spherical_crd(self):
        """Test parity for sphericalCRD method"""
        bad_chans = [3, 7]
        bad_chans_matlab = [4, 8]  # 1-based for MATLAB

        py_result = eeg_interp(self.test_EEG, bad_chans, method='sphericalCRD')
        ml_result = self.eeglab.eeg_interp(self.test_EEG, bad_chans_matlab, 'sphericalCRD')

        # Compare results using the helper method
        self._compare_eeg_results(py_result, ml_result)

    def test_parity_custom_params(self):
        """Test parity with custom parameters"""
        bad_chans = [1, 4]
        bad_chans_matlab = [2, 5]  # 1-based for MATLAB
        custom_params = (1e-6, 3, 10)

        py_result = eeg_interp(self.test_EEG, bad_chans, params=custom_params)

        # Convert tuple to numpy array for MATLAB
        custom_params_array = np.array(custom_params)
        ml_result = self.eeglab.eeg_interp(self.test_EEG, bad_chans_matlab, [], [], custom_params_array)

        # Compare results using the helper method
        self._compare_eeg_results(py_result, ml_result)

    def test_parity_custom_time_range(self):
        """Test parity with custom time range"""
        bad_chans = [2, 8]
        bad_chans_matlab = [3, 9]  # 1-based for MATLAB
        t_range = (-0.5, 0.5)

        py_result = eeg_interp(self.test_EEG, bad_chans, method='spherical', t_range=t_range)

        # Convert tuple to numpy array for MATLAB
        t_range_array = np.array(t_range)
        ml_result = self.eeglab.eeg_interp(self.test_EEG, bad_chans_matlab, 'spherical', t_range_array)

        # Compare results using the helper method
        self._compare_eeg_results(py_result, ml_result)

    def test_parity_multiple_trials(self):
        """Test parity with multiple trials (epochs)"""
        # Create multi-trial EEG data
        multi_trial_EEG = self.test_EEG.copy()
        multi_trial_EEG['trials'] = 3
        multi_trial_EEG['data'] = np.random.randn(32, 1000, 3) * 50

        bad_chans = [0, 5, 10]
        bad_chans_matlab = [1, 6, 11]  # 1-based for MATLAB

        py_result = eeg_interp(multi_trial_EEG, bad_chans, method='spherical')
        ml_result = self.eeglab.eeg_interp(multi_trial_EEG, bad_chans_matlab, 'spherical')

        # Compare results using the helper method
        self._compare_eeg_results(py_result, ml_result)


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestSphericalSplineParity(unittest.TestCase):
    """Test parity between Python spheric_spline and MATLAB spheric_spline"""

    def setUp(self):
        """Set up MATLAB interface and test data"""
        self.eeglab = get_eeglab('MAT')

        # Set up test electrode positions
        np.random.seed(42)
        n_good = 10
        n_bad = 3
        n_points = 100

        # Generate electrode positions on unit sphere
        xyz_good = np.random.randn(3, n_good)
        xyz_good /= np.linalg.norm(xyz_good, axis=0)

        xyz_bad = np.random.randn(3, n_bad)
        xyz_bad /= np.linalg.norm(xyz_bad, axis=0)

        self.xelec, self.yelec, self.zelec = xyz_good
        self.xbad, self.ybad, self.zbad = xyz_bad
        self.values = np.random.randn(n_good, n_points)
        self.params = (0, 4, 7)

    def test_parity_spheric_spline_different_params(self):
        """Test parity with different parameter sets"""
        param_sets = [
            (0, 4, 7),  # spherical
            (1e-8, 3, 50),  # sphericalKang
            (1e-5, 4, 100),  # sphericalCRD (reduced iterations for speed)
        ]

        for params in param_sets:
            with self.subTest(params=params):
                py_result = spheric_spline(
                    self.xelec, self.yelec, self.zelec, self.xbad, self.ybad, self.zbad, self.values, params
                )

                # Convert tuple to numpy array for MATLAB
                params_array = np.array(params)
                ml_result = self.eeglab.spheric_spline(
                    self.xelec, self.yelec, self.zelec, self.xbad, self.ybad, self.zbad, self.values, params_array
                )

                # Extract interpolated values (4th output from MATLAB)
                if isinstance(ml_result, (list, tuple)) and len(ml_result) >= 4:
                    ml_interpolated = ml_result[3]
                else:
                    ml_interpolated = ml_result

                self.assertEqual(py_result.shape, ml_interpolated.shape)
                self.assertTrue(np.allclose(py_result, ml_interpolated, atol=1e-10))


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestComputeGParity(unittest.TestCase):
    """Test parity between Python computeg and MATLAB computeg"""

    def setUp(self):
        """Set up MATLAB interface and test data"""
        self.eeglab = get_eeglab('MAT')

        self.x = np.array([0.1, 0.2, 0.3])
        self.y = np.array([0.4, 0.5, 0.6])
        self.z = np.array([0.7, 0.8, 0.9])
        self.xelec = np.array([0.0, 1.0])
        self.yelec = np.array([0.0, 0.0])
        self.zelec = np.array([1.0, 1.0])
        self.params = (0, 4, 7)

    def test_parity_computeg_different_params(self):
        """Test parity with different parameter values"""
        param_sets = [
            (0, 2, 5),
            (0, 4, 7),
            (1e-8, 3, 20),  # Reduced iterations for speed
        ]

        for params in param_sets:
            with self.subTest(params=params):
                py_result = computeg(self.x, self.y, self.z, self.xelec, self.yelec, self.zelec, params)

                # Convert tuple to numpy array for MATLAB
                params_array = np.array(params)
                ml_result = self.eeglab.computeg(
                    self.x, self.y, self.z, self.xelec, self.yelec, self.zelec, params_array
                )

                self.assertEqual(py_result.shape, ml_result.shape)
                self.assertTrue(np.allclose(py_result, ml_result, atol=1e-10))

        # The time range restoration code should have been executed
        # (lines 86-87 in the original code)
        # Note: Some values might be NaN or Inf for identical points, that's expected


class TestEegInterpChanlocs(unittest.TestCase):
    """Test the new chanloc structure functionality in eeg_interp"""

    def setUp(self):
        """Set up test EEG structure with known channel layout"""
        # Create a test EEG structure with 4 channels
        self.test_EEG = {
            'data': np.random.randn(4, 100, 1),  # 4 channels, 100 time points, 1 trial
            'nbchan': 4,
            'pnts': 100,
            'trials': 1,
            'srate': 500,
            'xmin': 0,
            'xmax': 0.2,
            'chanlocs': [
                {'labels': 'Fp1', 'X': 0.1, 'Y': 0.8, 'Z': 0.6},
                {'labels': 'Fp2', 'X': -0.1, 'Y': 0.8, 'Z': 0.6},
                {'labels': 'F3', 'X': 0.4, 'Y': 0.6, 'Z': 0.7},
                {'labels': 'F4', 'X': -0.4, 'Y': 0.6, 'Z': 0.7},
            ],
        }

        # Store original data for comparison
        self.original_data = self.test_EEG['data'].copy()

    def test_chanloc_no_overlap_appends_channels(self):
        """Test Case 2: No overlap should append new channels"""
        # Create completely new chanlocs with no overlap
        new_chanlocs = [
            {'labels': 'T7', 'X': 0.8, 'Y': 0.0, 'Z': 0.6},
            {'labels': 'T8', 'X': -0.8, 'Y': 0.0, 'Z': 0.6},
        ]

        result = eeg_interp(self.test_EEG.copy(), new_chanlocs)

        # Should have original 4 + 2 new = 6 channels
        self.assertEqual(result['nbchan'], 6)
        self.assertEqual(len(result['chanlocs']), 6)
        self.assertEqual(result['data'].shape, (6, 100, 1))

        # Original data should be preserved in first 4 channels
        np.testing.assert_array_equal(result['data'][:4, :, :], self.original_data)

        # New channels should have interpolated data (not zeros)
        new_channel_data = result['data'][4:, :, :]
        self.assertFalse(np.allclose(new_channel_data, 0))

        # Channel labels should be correct
        expected_labels = ['Fp1', 'Fp2', 'F3', 'F4', 'T7', 'T8']
        actual_labels = [ch['labels'] for ch in result['chanlocs']]
        self.assertEqual(actual_labels, expected_labels)

    def test_chanloc_partial_overlap_case(self):
        """Test partial overlap case (not subset, not disjoint)"""
        # Create chanlocs with partial overlap
        partial_overlap_chanlocs = [
            {'labels': 'Fp1', 'X': 0.1, 'Y': 0.8, 'Z': 0.6},  # exists
            {'labels': 'T7', 'X': 0.8, 'Y': 0.0, 'Z': 0.6},  # new
            {'labels': 'F3', 'X': 0.4, 'Y': 0.6, 'Z': 0.7},  # exists
        ]

        result = eeg_interp(self.test_EEG.copy(), partial_overlap_chanlocs)

        # Should interpolate the channels that exist (Fp1=0, F3=2)
        # Original data should be preserved, but interpolated channels should change
        self.assertEqual(result['nbchan'], 4)  # Original structure preserved
        self.assertEqual(result['data'].shape, (4, 100, 1))

        # Channels 0 (Fp1) and 2 (F3) should have been interpolated
        # Check that they're different from original (interpolated)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(result['data'][0, :, :], self.original_data[0, :, :])
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(result['data'][2, :, :], self.original_data[2, :, :])

        # Channels 1 (Fp2) and 3 (F4) should be preserved (not in bad_chans)
        np.testing.assert_array_equal(result['data'][1, :, :], self.original_data[1, :, :])
        np.testing.assert_array_equal(result['data'][3, :, :], self.original_data[3, :, :])

    def test_chanloc_with_multiple_trials(self):
        """Test chanloc functionality with multiple trials"""
        # Create multi-trial data
        multi_trial_EEG = self.test_EEG.copy()
        multi_trial_EEG['data'] = np.random.randn(4, 100, 3)  # 3 trials
        multi_trial_EEG['trials'] = 3
        original_multi_data = multi_trial_EEG['data'].copy()

        # Test superset case with multi-trial data
        superset_chanlocs = [
            {'labels': 'Fp1', 'X': 0.1, 'Y': 0.8, 'Z': 0.6},
            {'labels': 'Fp2', 'X': -0.1, 'Y': 0.8, 'Z': 0.6},
            {'labels': 'F3', 'X': 0.4, 'Y': 0.6, 'Z': 0.7},
            {'labels': 'F4', 'X': -0.4, 'Y': 0.6, 'Z': 0.7},
            {'labels': 'C3', 'X': 0.6, 'Y': 0.0, 'Z': 0.8},
        ]

        result = eeg_interp(multi_trial_EEG, superset_chanlocs)

        # Should handle 3D data correctly
        self.assertEqual(result['data'].shape, (5, 100, 3))
        self.assertEqual(result['nbchan'], 5)

        # Original data should be preserved in correct positions
        np.testing.assert_array_equal(result['data'][:4, :, :], original_multi_data)


if __name__ == '__main__':
    unittest.main()
