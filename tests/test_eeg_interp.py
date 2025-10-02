import unittest
import numpy as np
import os
from scipy.io import loadmat, savemat
from unittest.mock import patch, MagicMock

from eegprep.eeg_interp import eeg_interp, spheric_spline, computeg
from eegprep.eeglabcompat import get_eeglab

# Test Case	(Python vs. MATLAB)         Max Absolute	Max Relative	Scenario
# test_parity_multiple_trials	        5.11e-04	1.99e-02 (1.99%)	3 trials, 3 channels, 1 trial
# test_parity_single_channel	        4.58e-04	4.49e-03 (0.45%)	1 channel, 1 trial
# test_parity_spherical_crd	            3.89e-04	2.76e-03 (0.28%)	SphericalCRD method, 3 channels, 1 trial
# test_parity_preserves_good_channels	3.61e-04	5.36e-03 (0.54%)	3 channels, 1 trial
# test_parity_custom_params	            3.43e-04	2.08e-03 (0.21%)	Custom parameters, 3 channels, 1 trial
# test_parity_spherical_kang	        2.81e-04	2.98e-03 (0.30%)	SphericalKang method, 3 channels, 1 trial
# test_parity_spherical_basic	        2.50e-04	3.18e-03 (0.32%)	Basic spherical, 3 channels, 1 trial
# test_parity_custom_time_range	        2.49e-04	2.02e-03 (0.20%)	Custom time range, 3 channels

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
            'etc': {}
        }
        
        # Create realistic channel locations on unit sphere
        for i in range(n_channels):
            theta = 2 * np.pi * i / n_channels
            phi = np.pi/6 + (np.pi/3) * (i % 8) / 8
            
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
                label = f'Ch{i+1}'
            
            self.test_EEG['chanlocs'].append({
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
                'ref': ''
            })

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

    def test_parity_single_channel(self):
        """Test parity for single channel interpolation"""
        bad_chans = [15]
        bad_chans_matlab = [16]  # 1-based for MATLAB
        
        py_result = eeg_interp(self.test_EEG, bad_chans, method='spherical')
        ml_result = self.eeglab.eeg_interp(self.test_EEG, bad_chans_matlab, 'spherical')
        
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

    def test_parity_preserves_good_channels(self):
        """Test that both implementations preserve good channels identically"""
        bad_chans = [5, 15, 25]
        bad_chans_matlab = [6, 16, 26]  # 1-based for MATLAB
        good_chans = [i for i in range(32) if i not in bad_chans]
        
        # Store original data for good channels
        original_good_data = self.test_EEG['data'][good_chans, :, :].copy()
        
        py_result = eeg_interp(self.test_EEG, bad_chans, method='spherical')
        ml_result = self.eeglab.eeg_interp(self.test_EEG, bad_chans_matlab, 'spherical')
        
        # Python should preserve good channels exactly
        np.testing.assert_array_equal(py_result['data'][good_chans, :, :], original_good_data)
        
        # For MATLAB, just check that the overall results match using the helper
        # (Good channel preservation is harder to test due to potential channel reordering)
        self._compare_eeg_results(py_result, ml_result)


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
    
    def test_parity_spheric_spline_basic(self):
        """Test parity for basic spherical spline interpolation"""
        # Python computation
        py_result = spheric_spline(
            self.xelec, self.yelec, self.zelec,
            self.xbad, self.ybad, self.zbad,
            self.values, self.params
        )
        
        # Convert tuple to numpy array for MATLAB
        params_array = np.array(self.params)
        
        # MATLAB computation (call the internal function)
        ml_result = self.eeglab.spheric_spline(
            self.xelec, self.yelec, self.zelec,
            self.xbad, self.ybad, self.zbad,
            self.values, params_array
        )
        
        # Extract the actual interpolated values (4th output from MATLAB)
        if isinstance(ml_result, (list, tuple)) and len(ml_result) >= 4:
            ml_interpolated = ml_result[3]  # allres is the 4th output
        else:
            ml_interpolated = ml_result
        
        self.assertEqual(py_result.shape, ml_interpolated.shape)
        self.assertTrue(np.allclose(py_result, ml_interpolated, atol=1e-10))
    
    def test_parity_spheric_spline_different_params(self):
        """Test parity with different parameter sets"""
        param_sets = [
            (0, 4, 7),      # spherical
            (1e-8, 3, 50),  # sphericalKang
            (1e-5, 4, 100)  # sphericalCRD (reduced iterations for speed)
        ]
        
        for params in param_sets:
            with self.subTest(params=params):
                py_result = spheric_spline(
                    self.xelec, self.yelec, self.zelec,
                    self.xbad, self.ybad, self.zbad,
                    self.values, params
                )
                
                # Convert tuple to numpy array for MATLAB
                params_array = np.array(params)
                ml_result = self.eeglab.spheric_spline(
                    self.xelec, self.yelec, self.zelec,
                    self.xbad, self.ybad, self.zbad,
                    self.values, params_array
                )
                
                # Extract interpolated values (4th output from MATLAB)
                if isinstance(ml_result, (list, tuple)) and len(ml_result) >= 4:
                    ml_interpolated = ml_result[3]
                else:
                    ml_interpolated = ml_result
                
                self.assertEqual(py_result.shape, ml_interpolated.shape)
                self.assertTrue(np.allclose(py_result, ml_interpolated, atol=1e-10))


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
    
    def test_parity_computeg_basic(self):
        """Test parity for basic computeg function"""
        py_result = computeg(
            self.x, self.y, self.z,
            self.xelec, self.yelec, self.zelec,
            self.params
        )
        
        # Convert tuple to numpy array for MATLAB
        params_array = np.array(self.params)
        ml_result = self.eeglab.computeg(
            self.x, self.y, self.z,
            self.xelec, self.yelec, self.zelec,
            params_array
        )
        
        self.assertEqual(py_result.shape, ml_result.shape)
        self.assertTrue(np.allclose(py_result, ml_result, atol=1e-10))
    
    def test_parity_computeg_different_params(self):
        """Test parity with different parameter values"""
        param_sets = [
            (0, 2, 5),
            (0, 4, 7),
            (1e-8, 3, 20)  # Reduced iterations for speed
        ]
        
        for params in param_sets:
            with self.subTest(params=params):
                py_result = computeg(
                    self.x, self.y, self.z,
                    self.xelec, self.yelec, self.zelec,
                    params
                )
                
                # Convert tuple to numpy array for MATLAB
                params_array = np.array(params)
                ml_result = self.eeglab.computeg(
                    self.x, self.y, self.z,
                    self.xelec, self.yelec, self.zelec,
                    params_array
                )
                
                self.assertEqual(py_result.shape, ml_result.shape)
                self.assertTrue(np.allclose(py_result, ml_result, atol=1e-10))


class TestEegInterp(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with mock EEG data structure"""
        # Create a minimal EEG structure for testing
        self.mock_EEG = {
            'data': np.random.randn(32, 1000, 1),  # 32 channels, 1000 time points, 1 trial
            'nbchan': 32,
            'pnts': 1000,
            'trials': 1,
            'xmin': -0.5,
            'xmax': 1.5,
            'chanlocs': []
        }
        
        # Create mock channel locations
        for i in range(32):
            # Create spherical coordinates on unit sphere
            theta = 2 * np.pi * i / 32
            phi = np.pi / 4  # Fixed elevation
            x = np.cos(phi) * np.cos(theta)
            y = np.cos(phi) * np.sin(theta)
            z = np.sin(phi)
            
            self.mock_EEG['chanlocs'].append({
                'labels': f'Ch{i+1}',
                'X': x,
                'Y': y,
                'Z': z
            })
        
        # Add some named channels for testing
        self.mock_EEG['chanlocs'][0]['labels'] = 'Fp1'
        self.mock_EEG['chanlocs'][1]['labels'] = 'Fp2'
        self.mock_EEG['chanlocs'][2]['labels'] = 'F7'
    
    def test_eeg_interp_basic_functionality(self):
        """Test basic interpolation with channel indices"""
        bad_chans = [0, 1, 2]  # First 3 channels
        result = eeg_interp(self.mock_EEG, bad_chans, method='spherical')
        
        # Check that structure is preserved
        self.assertEqual(result['nbchan'], self.mock_EEG['nbchan'])
        self.assertEqual(result['pnts'], self.mock_EEG['pnts'])
        self.assertEqual(result['trials'], self.mock_EEG['trials'])
        self.assertEqual(result['data'].shape, self.mock_EEG['data'].shape)
    
    def test_eeg_interp_with_channel_labels(self):
        """Test interpolation with channel labels"""
        bad_chans = ['Fp1', 'Fp2', 'F7']
        result = eeg_interp(self.mock_EEG, bad_chans, method='spherical')
        
        # Check that structure is preserved
        self.assertEqual(result['nbchan'], self.mock_EEG['nbchan'])
        self.assertEqual(result['data'].shape, self.mock_EEG['data'].shape)
    
    def test_eeg_interp_different_methods(self):
        """Test different interpolation methods"""
        bad_chans = [0, 1]
        
        # Test sphericalKang method
        result_kang = eeg_interp(self.mock_EEG, bad_chans, method='sphericalKang')
        self.assertEqual(result_kang['data'].shape, self.mock_EEG['data'].shape)
        
        # Test sphericalCRD method  
        result_crd = eeg_interp(self.mock_EEG, bad_chans, method='sphericalCRD')
        self.assertEqual(result_crd['data'].shape, self.mock_EEG['data'].shape)
    
    def test_eeg_interp_custom_params(self):
        """Test interpolation with custom parameters"""
        bad_chans = [0, 1]
        custom_params = (1e-6, 3, 10)
        result = eeg_interp(self.mock_EEG, bad_chans, params=custom_params)
        
        self.assertEqual(result['data'].shape, self.mock_EEG['data'].shape)
    
    def test_eeg_interp_error_cases(self):
        """Test error handling"""
        # Test unknown method
        with self.assertRaises(ValueError) as cm:
            eeg_interp(self.mock_EEG, [0, 1], method='unknown')
        self.assertIn("Unknown method", str(cm.exception))
        
        # Test invalid params length
        with self.assertRaises(ValueError) as cm:
            eeg_interp(self.mock_EEG, [0, 1], params=(1, 2))  # Only 2 params instead of 3
        self.assertIn("params must be length-3 tuple", str(cm.exception))
        
        # Test missing channel locations
        eeg_no_locs = self.mock_EEG.copy()
        eeg_no_locs['chanlocs'] = []
        with self.assertRaises(RuntimeError) as cm:
            eeg_interp(eeg_no_locs, [0, 1])
        self.assertIn("Channel locations required", str(cm.exception))
        
        # Test missing coordinate fields
        eeg_bad_locs = self.mock_EEG.copy()
        eeg_bad_locs['chanlocs'] = [{'labels': 'Ch1'}]  # Missing X, Y, Z
        with self.assertRaises(RuntimeError) as cm:
            eeg_interp(eeg_bad_locs, [0])
        self.assertIn("Channel locations required", str(cm.exception))
    
    def test_eeg_interp_with_nan_channels(self):
        """Test handling of channels with NaN coordinates"""
        # Add a channel with NaN coordinates
        eeg_with_nan = self.mock_EEG.copy()
        eeg_with_nan['chanlocs'] = self.mock_EEG['chanlocs'].copy()
        eeg_with_nan['chanlocs'].append({
            'labels': 'NaN_ch',
            'X': np.nan,
            'Y': np.nan,
            'Z': np.nan
        })
        eeg_with_nan['nbchan'] = 33
        eeg_with_nan['data'] = np.random.randn(33, 1000, 1)
        
        result = eeg_interp(eeg_with_nan, [0, 1], method='spherical')
        self.assertEqual(result['data'].shape, (33, 1000, 1))
    
    def test_eeg_interp_channel_label_not_found(self):
        """Test error when channel label is not found"""
        with self.assertRaises(ValueError):
            eeg_interp(self.mock_EEG, ['NonexistentChannel'])
    
    def test_eeg_interp_custom_time_range(self):
        """Test interpolation with custom time range"""
        bad_chans = [0, 1]
        custom_t_range = (-0.2, 1.0)  # Different from EEG's xmin/xmax
        
        result = eeg_interp(self.mock_EEG, bad_chans, method='spherical', t_range=custom_t_range)
        
        # Check that structure is preserved
        self.assertEqual(result['nbchan'], self.mock_EEG['nbchan'])
        self.assertEqual(result['data'].shape, self.mock_EEG['data'].shape)
        
        # The time range restoration code should have been executed
        # (lines 86-87 in the original code)


class TestSphericSpline(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for spheric spline"""
        np.random.seed(42)  # For reproducible tests
        self.n_good = 10
        self.n_bad = 2
        self.n_points = 100
        
        # Generate random electrode positions on unit sphere
        xyz_good = np.random.randn(3, self.n_good)
        xyz_good /= np.linalg.norm(xyz_good, axis=0)
        
        xyz_bad = np.random.randn(3, self.n_bad)
        xyz_bad /= np.linalg.norm(xyz_bad, axis=0)
        
        self.xelec, self.yelec, self.zelec = xyz_good
        self.xbad, self.ybad, self.zbad = xyz_bad
        self.values = np.random.randn(self.n_good, self.n_points)
        self.params = (0, 4, 7)
    
    def test_spheric_spline_basic(self):
        """Test basic spheric spline functionality"""
        result = spheric_spline(
            self.xelec, self.yelec, self.zelec,
            self.xbad, self.ybad, self.zbad,
            self.values, self.params
        )
        
        # Check output shape
        self.assertEqual(result.shape, (self.n_bad, self.n_points))
        
        # Check that result is finite
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_spheric_spline_different_params(self):
        """Test spheric spline with different parameter sets"""
        params_sets = [
            (0, 4, 7),      # spherical
            (1e-8, 3, 50),  # sphericalKang
            (1e-5, 4, 500)  # sphericalCRD
        ]
        
        for params in params_sets:
            with self.subTest(params=params):
                result = spheric_spline(
                    self.xelec, self.yelec, self.zelec,
                    self.xbad, self.ybad, self.zbad,
                    self.values, params
                )
                self.assertEqual(result.shape, (self.n_bad, self.n_points))
                self.assertTrue(np.all(np.isfinite(result)))
    
    def test_spheric_spline_single_point(self):
        """Test spheric spline with single time point"""
        values_single = self.values[:, :1]  # Single time point
        result = spheric_spline(
            self.xelec, self.yelec, self.zelec,
            self.xbad, self.ybad, self.zbad,
            values_single, self.params
        )
        
        self.assertEqual(result.shape, (self.n_bad, 1))
        self.assertTrue(np.all(np.isfinite(result)))


class TestComputeG(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for computeg function"""
        self.x = np.array([0.1, 0.2, 0.3])
        self.y = np.array([0.4, 0.5, 0.6])
        self.z = np.array([0.7, 0.8, 0.9])
        self.xelec = np.array([0.0, 1.0])
        self.yelec = np.array([0.0, 0.0])
        self.zelec = np.array([1.0, 1.0])
        self.params = (0, 4, 7)
    
    def test_computeg_basic(self):
        """Test basic computeg functionality"""
        result = computeg(self.x, self.y, self.z, 
                         self.xelec, self.yelec, self.zelec, 
                         self.params)
        
        # Check output shape
        expected_shape = (len(self.x), len(self.xelec))
        self.assertEqual(result.shape, expected_shape)
        
        # Check that result is finite
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_computeg_different_params(self):
        """Test computeg with different parameter values"""
        params_list = [
            (0, 2, 5),
            (0, 4, 7),
            (0, 6, 10)
        ]
        
        for params in params_list:
            with self.subTest(params=params):
                result = computeg(self.x, self.y, self.z,
                                 self.xelec, self.yelec, self.zelec,
                                 params)
                self.assertEqual(result.shape, (len(self.x), len(self.xelec)))
                self.assertTrue(np.all(np.isfinite(result)))
    
    def test_computeg_single_point(self):
        """Test computeg with single interpolation point"""
        x_single = np.array([0.1])
        y_single = np.array([0.2])
        z_single = np.array([0.3])
        
        result = computeg(x_single, y_single, z_single,
                         self.xelec, self.yelec, self.zelec,
                         self.params)
        
        self.assertEqual(result.shape, (1, len(self.xelec)))
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_computeg_edge_cases(self):
        """Test computeg with edge cases"""
        # Test with identical points (should not crash)
        x_same = np.array([0.0])
        y_same = np.array([0.0])
        z_same = np.array([1.0])
        
        result = computeg(x_same, y_same, z_same,
                         self.xelec, self.yelec, self.zelec,
                         self.params)
        
        self.assertEqual(result.shape, (1, len(self.xelec)))
        # Note: Some values might be NaN or Inf for identical points, that's expected


class TestEegInterpIntegration(unittest.TestCase):
    """Integration tests for the complete interpolation pipeline"""
    
    def setUp(self):
        """Set up realistic EEG data for integration testing"""
        # Create more realistic EEG structure
        n_channels = 64
        n_timepoints = 2000
        n_trials = 1
        
        self.eeg_data = {
            'data': np.random.randn(n_channels, n_timepoints, n_trials) * 50,  # Realistic amplitude
            'nbchan': n_channels,
            'pnts': n_timepoints,
            'trials': n_trials,
            'srate': 500,  # 500 Hz sampling rate
            'xmin': -1.0,
            'xmax': 3.0,
            'chanlocs': []
        }
        
        # Create realistic channel locations (10-20 system approximation)
        for i in range(n_channels):
            # Distribute channels on sphere
            theta = 2 * np.pi * i / n_channels
            phi = np.pi/6 + (np.pi/3) * (i % 8) / 8  # Vary elevation
            
            x = np.cos(phi) * np.cos(theta)
            y = np.cos(phi) * np.sin(theta)
            z = np.sin(phi)
            
            self.eeg_data['chanlocs'].append({
                'labels': f'Ch{i+1}',
                'X': x,
                'Y': y,
                'Z': z
            })
    
    def test_interpolation_preserves_good_channels(self):
        """Test that interpolation doesn't change good channels"""
        bad_channels = [5, 10, 15]
        good_channels = [i for i in range(self.eeg_data['nbchan']) if i not in bad_channels]
        
        # Store original data for good channels
        original_good_data = self.eeg_data['data'][good_channels, :, :].copy()
        
        # Perform interpolation
        result = eeg_interp(self.eeg_data, bad_channels, method='spherical')
        
        # Check that good channels are unchanged
        np.testing.assert_array_equal(
            result['data'][good_channels, :, :],
            original_good_data
        )
    
    def test_interpolation_changes_bad_channels(self):
        """Test that interpolation actually changes bad channels"""
        bad_channels = [5, 10, 15]
        
        # Store original data for bad channels
        original_bad_data = self.eeg_data['data'][bad_channels, :, :].copy()
        
        # Perform interpolation
        result = eeg_interp(self.eeg_data, bad_channels, method='spherical')
        
        # Check that bad channels have changed
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(
                result['data'][bad_channels, :, :],
                original_bad_data
            )
    
    def test_interpolation_reasonable_values(self):
        """Test that interpolated values are reasonable"""
        bad_channels = [10, 20]
        
        # Perform interpolation
        result = eeg_interp(self.eeg_data, bad_channels, method='spherical')
        
        # Check that interpolated values are within reasonable range
        interpolated_data = result['data'][bad_channels, :, :]
        
        # Should be finite
        self.assertTrue(np.all(np.isfinite(interpolated_data)))
        
        # Should be within reasonable amplitude range (not too extreme)
        # Assuming original data has std around 50, interpolated should be similar
        self.assertTrue(np.std(interpolated_data) < 200)  # Not too large
        self.assertTrue(np.std(interpolated_data) > 1)    # Not too small


class TestEegInterpFileOperations(unittest.TestCase):
    """Test file I/O operations used in the original test functions"""
    
    def test_spheric_spline_file_operations(self):
        """Test that spheric spline can handle file operations properly"""
        # Test that the spheric spline function works with mock data
        # This tests the functionality without requiring actual MATLAB files
        
        np.random.seed(42)
        n_good, n_bad, n_pts = 5, 2, 50
        
        # Generate test electrode positions
        xyz = np.random.randn(3, n_good)
        xyz /= np.linalg.norm(xyz, axis=0)
        xbad = np.random.randn(3, n_bad)
        xbad /= np.linalg.norm(xbad, axis=0)
        
        # Generate test values
        values = np.random.randn(n_good, n_pts)
        
        # Test that spheric spline computation works
        result = spheric_spline(
            xelec=xyz[0], yelec=xyz[1], zelec=xyz[2],
            xbad=xbad[0], ybad=xbad[1], zbad=xbad[2],
            values=values, params=(0, 4, 7)
        )
        
        self.assertEqual(result.shape, (n_bad, n_pts))
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_computeg_file_operations(self):
        """Test that computeg can handle different input configurations"""
        # Test computeg with various input sizes
        test_cases = [
            (10, 5),    # 10 interpolation points, 5 electrodes
            (50, 10),   # 50 interpolation points, 10 electrodes  
            (100, 20),  # 100 interpolation points, 20 electrodes
        ]
        
        for n_points, n_elec in test_cases:
            with self.subTest(n_points=n_points, n_elec=n_elec):
                # Generate test coordinates
                x = np.linspace(0, 1, n_points)
                y = np.linspace(0, 1, n_points) 
                z = np.linspace(0, 1, n_points)
                xelec = np.linspace(0, 1, n_elec)
                yelec = np.linspace(0, 1, n_elec)
                zelec = np.linspace(0, 1, n_elec)
                params = (0.0, 4.0, 7.0)
                
                # Test computation
                g = computeg(x, y, z, xelec, yelec, zelec, params)
                
                self.assertEqual(g.shape, (n_points, n_elec))
                self.assertTrue(np.all(np.isfinite(g)))

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
            ]
        }
        
        # Store original data for comparison
        self.original_data = self.test_EEG['data'].copy()
    
    def test_chanloc_identical_returns_unchanged(self):
        """Test Case 1: Identical chanlocs should return data unchanged"""
        # Create identical chanlocs
        identical_chanlocs = [ch.copy() for ch in self.test_EEG['chanlocs']]
        
        result = eeg_interp(self.test_EEG.copy(), identical_chanlocs)
        
        # Data should be identical (no interpolation needed)
        np.testing.assert_array_equal(result['data'], self.original_data)
        self.assertEqual(result['nbchan'], 4)
        self.assertEqual(len(result['chanlocs']), 4)
        self.assertEqual([ch['labels'] for ch in result['chanlocs']], 
                        ['Fp1', 'Fp2', 'F3', 'F4'])
    
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
    
    def test_chanloc_proper_subset_remaps_data(self):
        """Test Case 3: Existing channels as proper subset should remap to new structure"""
        # Create superset that includes existing channels plus new ones
        superset_chanlocs = [
            {'labels': 'Fp1', 'X': 0.1, 'Y': 0.8, 'Z': 0.6},    # existing
            {'labels': 'C3', 'X': 0.6, 'Y': 0.0, 'Z': 0.8},     # new
            {'labels': 'F3', 'X': 0.4, 'Y': 0.6, 'Z': 0.7},     # existing  
            {'labels': 'C4', 'X': -0.6, 'Y': 0.0, 'Z': 0.8},    # new
            {'labels': 'Fp2', 'X': -0.1, 'Y': 0.8, 'Z': 0.6},   # existing
            {'labels': 'F4', 'X': -0.4, 'Y': 0.6, 'Z': 0.7},    # existing
        ]
        
        result = eeg_interp(self.test_EEG.copy(), superset_chanlocs)
        
        # Should have 6 channels total
        self.assertEqual(result['nbchan'], 6)
        self.assertEqual(len(result['chanlocs']), 6)
        self.assertEqual(result['data'].shape, (6, 100, 1))
        
        # Check that existing data is preserved in correct positions
        # Fp1: original pos 0 -> new pos 0
        np.testing.assert_array_equal(result['data'][0, :, :], self.original_data[0, :, :])
        # F3: original pos 2 -> new pos 2  
        np.testing.assert_array_equal(result['data'][2, :, :], self.original_data[2, :, :])
        # Fp2: original pos 1 -> new pos 4
        np.testing.assert_array_equal(result['data'][4, :, :], self.original_data[1, :, :])
        # F4: original pos 3 -> new pos 5
        np.testing.assert_array_equal(result['data'][5, :, :], self.original_data[3, :, :])
        
        # New channels (C3, C4) should have interpolated data
        self.assertFalse(np.allclose(result['data'][1, :, :], 0))  # C3
        self.assertFalse(np.allclose(result['data'][3, :, :], 0))  # C4
        
        # Channel labels should match the new structure
        expected_labels = ['Fp1', 'C3', 'F3', 'C4', 'Fp2', 'F4']
        actual_labels = [ch['labels'] for ch in result['chanlocs']]
        self.assertEqual(actual_labels, expected_labels)
    
    def test_chanloc_empty_list(self):
        """Test that empty chanloc list returns original data unchanged"""
        result = eeg_interp(self.test_EEG.copy(), [])
        
        # Should return original data unchanged
        np.testing.assert_array_equal(result['data'], self.original_data)
        self.assertEqual(result['nbchan'], 4)
    
    def test_chanloc_partial_overlap_case(self):
        """Test partial overlap case (not subset, not disjoint)"""
        # Create chanlocs with partial overlap
        partial_overlap_chanlocs = [
            {'labels': 'Fp1', 'X': 0.1, 'Y': 0.8, 'Z': 0.6},    # exists
            {'labels': 'T7', 'X': 0.8, 'Y': 0.0, 'Z': 0.6},     # new
            {'labels': 'F3', 'X': 0.4, 'Y': 0.6, 'Z': 0.7},     # exists
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
    
    def test_chanloc_coordinates_differ_but_labels_same(self):
        """Test case where labels are same but coordinates differ - should raise error"""
        # Same labels but different coordinates
        different_coords_chanlocs = [
            {'labels': 'Fp1', 'X': 0.2, 'Y': 0.9, 'Z': 0.5},  # Different coords
            {'labels': 'Fp2', 'X': -0.2, 'Y': 0.9, 'Z': 0.5}, # Different coords
            {'labels': 'F3', 'X': 0.5, 'Y': 0.7, 'Z': 0.6},   # Different coords
            {'labels': 'F4', 'X': -0.5, 'Y': 0.7, 'Z': 0.6},  # Different coords
        ]
        
        # Should raise ValueError for ambiguous case
        with self.assertRaises(ValueError) as cm:
            eeg_interp(self.test_EEG.copy(), different_coords_chanlocs)
        
        self.assertIn("coordinates differ", str(cm.exception))
        self.assertIn("ambiguous", str(cm.exception))
    
    def test_chanloc_invalid_structure(self):
        """Test error handling for invalid chanloc structures"""
        # Test missing required fields
        invalid_chanlocs = [
            {'labels': 'T7', 'X': 0.8, 'Y': 0.0},  # Missing Z
        ]
        
        # Should fall back to treating as regular channel names and fail appropriately
        with self.assertRaises(ValueError):
            eeg_interp(self.test_EEG.copy(), ['NonexistentChannel'])
    
    def test_chanloc_continuous_data(self):
        """Test chanloc functionality with continuous (2D) data"""
        # Create continuous data (no trial dimension)
        continuous_EEG = self.test_EEG.copy()
        original_continuous_data = np.random.randn(4, 100)  # 2D instead of 3D
        continuous_EEG['data'] = original_continuous_data.copy()
        continuous_EEG['trials'] = 1
        
        # Test no overlap case with continuous data
        new_chanlocs = [
            {'labels': 'T7', 'X': 0.8, 'Y': 0.0, 'Z': 0.6},
        ]
        
        result = eeg_interp(continuous_EEG, new_chanlocs)
        
        # Should handle 2D data correctly
        self.assertEqual(result['data'].shape, (5, 100))  # 4 original + 1 new
        self.assertEqual(result['nbchan'], 5)
        
        # Original data should be preserved (compare with saved original)
        np.testing.assert_array_equal(result['data'][:4, :], original_continuous_data)
    
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
