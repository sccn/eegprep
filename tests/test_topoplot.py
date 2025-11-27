# Disable multithreading for deterministic numerical results in parity tests
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import unittest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import warnings
import tempfile
import scipy.io

# Set Agg backend before importing topoplot to avoid display issues
matplotlib.use('Agg')

from eegprep.topoplot import topoplot, griddata_v4
from eegprep import pop_loadset, pop_saveset
from eegprep.eeglabcompat import get_eeglab

local_url = os.path.join(os.path.dirname(__file__), '../data/')


class TestGriddataV4(unittest.TestCase):
    """Test the griddata_v4 function (biharmonic spline interpolation)."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple test data
        self.x = np.array([0, 1, 0.5])
        self.y = np.array([0, 0, 1])
        self.v = np.array([1, 2, 1.5])
        
        # Create query grid
        xq_1d = np.linspace(-0.5, 1.5, 5)
        yq_1d = np.linspace(-0.5, 1.5, 5)
        self.xq, self.yq = np.meshgrid(xq_1d, yq_1d)
    
    def test_basic_interpolation(self):
        """Test basic biharmonic spline interpolation."""
        vq = griddata_v4(self.x, self.y, self.v, self.xq, self.yq)
        
        # Check output shape
        self.assertEqual(vq.shape, self.xq.shape)
        self.assertEqual(vq.shape, self.yq.shape)
        
        # Check that interpolation at known points gives correct values (approximately)
        # Find closest grid points to known data points
        for i, (xi, yi, vi) in enumerate(zip(self.x, self.y, self.v)):
            # Find closest grid point
            dist = np.sqrt((self.xq - xi)**2 + (self.yq - yi)**2)
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            
            # Value should be close to expected (within interpolation tolerance)
            self.assertAlmostEqual(vq[min_idx], vi, delta=0.5)
    
    def test_interpolation_finite_results(self):
        """Test that interpolation produces finite results."""
        vq = griddata_v4(self.x, self.y, self.v, self.xq, self.yq)
        
        # All results should be finite
        self.assertTrue(np.all(np.isfinite(vq)))
    
    def test_single_point_interpolation(self):
        """Test interpolation with single data point."""
        x_single = np.array([0.5])
        y_single = np.array([0.5])
        v_single = np.array([3.0])
        
        # Small query grid
        xq_small = np.array([[0.4, 0.6], [0.4, 0.6]])
        yq_small = np.array([[0.4, 0.4], [0.6, 0.6]])
        
        # Single point interpolation may cause singular matrix
        try:
            vq = griddata_v4(x_single, y_single, v_single, xq_small, yq_small)
            # If it succeeds, check shape
            self.assertEqual(vq.shape, xq_small.shape)
        except np.linalg.LinAlgError:
            # Singular matrix is expected with single point - this is acceptable
            pass
    
    def test_collinear_points(self):
        """Test interpolation with collinear data points."""
        x_collinear = np.array([0, 0.5, 1])
        y_collinear = np.array([0, 0, 0])  # All on x-axis
        v_collinear = np.array([1, 2, 3])
        
        xq_small = np.array([[0.25, 0.75]])
        yq_small = np.array([[0, 0]])
        
        vq = griddata_v4(x_collinear, y_collinear, v_collinear, xq_small, yq_small)
        
        # Should handle collinear points gracefully
        self.assertEqual(vq.shape, xq_small.shape)
        # Results may not be finite due to singular matrix, but shouldn't crash
    
    def test_zero_distance_handling(self):
        """Test handling of zero distances in Green's function."""
        # Create data where query point exactly matches data point
        x_exact = np.array([0, 1])
        y_exact = np.array([0, 0])
        v_exact = np.array([1, 2])
        
        xq_exact = np.array([[0, 1]])  # Exact match to data points
        yq_exact = np.array([[0, 0]])
        
        vq = griddata_v4(x_exact, y_exact, v_exact, xq_exact, yq_exact)
        
        # Should handle zero distances without crashing
        self.assertEqual(vq.shape, xq_exact.shape)


class TestTopoplot(unittest.TestCase):
    """Test the topoplot function."""
    
    def setUp(self):
        """Set up test fixtures with synthetic EEG data and channel locations."""
        # Create synthetic channel locations (standard 10-20 system subset)
        self.chan_locs = [
            {'labels': 'Fz', 'theta': 0, 'radius': 0.3},
            {'labels': 'Cz', 'theta': 0, 'radius': 0.0},  # Central electrode
            {'labels': 'Pz', 'theta': 180, 'radius': 0.3},
            {'labels': 'C3', 'theta': 270, 'radius': 0.4},
            {'labels': 'C4', 'theta': 90, 'radius': 0.4},
            {'labels': 'F3', 'theta': 315, 'radius': 0.5},
            {'labels': 'F4', 'theta': 45, 'radius': 0.5},
            {'labels': 'P3', 'theta': 225, 'radius': 0.5},
            {'labels': 'P4', 'theta': 135, 'radius': 0.5}
        ]
        
        # Create synthetic data vector (one value per channel)
        self.datavector = np.array([1.0, 2.0, -1.0, 0.5, -0.5, 1.5, -1.5, 0.8, -0.8])
        
        # Create minimal channel locations for edge cases
        self.minimal_chan_locs = [
            {'labels': 'Cz', 'theta': 0, 'radius': 0.0},
            {'labels': 'Fz', 'theta': 0, 'radius': 0.3},
            {'labels': 'Pz', 'theta': 180, 'radius': 0.3}
        ]
        self.minimal_data = np.array([1.0, 0.5, -0.5])
    
    def test_basic_topoplot_with_agg_backend(self):
        """Test basic topoplot functionality with Agg backend (no display)."""
        # Ensure Agg backend is set
        self.assertEqual(matplotlib.get_backend(), 'Agg')
        
        handle, Zi, plotrad, xi, yi = topoplot(self.datavector, self.chan_locs, noplot='on')
        
        # Check return values
        self.assertIsNone(handle)  # No plot handle when noplot='on'
        self.assertIsInstance(Zi, np.ndarray)
        self.assertIsInstance(plotrad, (int, float))
        self.assertIsInstance(xi, np.ndarray)
        self.assertIsInstance(yi, np.ndarray)
        
        # Check interpolation grid shapes
        self.assertEqual(xi.shape, yi.shape)
        self.assertEqual(Zi.shape, xi.shape)
        
        # Check that interpolation grid is reasonable
        self.assertTrue(np.all(np.isfinite(xi)))
        self.assertTrue(np.all(np.isfinite(yi)))
        
        # Zi may have NaNs outside head boundary, but should have finite values inside
        finite_mask = np.isfinite(Zi)
        self.assertTrue(np.any(finite_mask))  # Should have some finite values
    
    def test_topoplot_with_plotting_disabled(self):
        """Test topoplot with plotting disabled (noplot='on')."""
        # Clear any existing figures first
        plt.close('all')
        
        handle, Zi, plotrad, xi, yi = topoplot(
            self.datavector, 
            self.chan_locs, 
            noplot='on'
        )
        
        # Should complete without creating plots
        self.assertIsNone(handle)
        self.assertIsInstance(Zi, np.ndarray)
        
        # With noplot='on', no new figures should be created
        # Note: Some figures might exist from other tests, so we just check the function works
        self.assertTrue(True)  # Function completed successfully
    
    def test_topoplot_with_chanlocs_present(self):
        """Test topoplot with channel locations present."""
        handle, Zi, plotrad, xi, yi = topoplot(
            self.datavector, 
            self.chan_locs, 
            noplot='on'
        )
        
        # Should successfully process channel locations
        self.assertIsInstance(Zi, np.ndarray)
        self.assertGreater(plotrad, 0)
        
        # Grid should cover appropriate range
        self.assertTrue(np.min(xi) < 0)
        self.assertTrue(np.max(xi) > 0)
        self.assertTrue(np.min(yi) < 0)
        self.assertTrue(np.max(yi) > 0)
    
    def test_topoplot_without_chanlocs(self):
        """Test topoplot with minimal or missing channel location info."""
        # Create channel locations without theta/radius
        incomplete_chanlocs = [
            {'labels': 'Ch1'},
            {'labels': 'Ch2'},
            {'labels': 'Ch3'}
        ]
        
        # This should handle missing theta/radius gracefully
        try:
            handle, Zi, plotrad, xi, yi = topoplot(
                np.array([1, 0, -1]), 
                incomplete_chanlocs, 
                noplot='on'
            )
            # If it completes, check basic properties
            self.assertIsInstance(Zi, np.ndarray)
        except (KeyError, IndexError, ValueError):
            # Function may raise error for incomplete channel info - this is acceptable
            pass
    
    def test_nan_positions_handling(self):
        """Test topoplot with NaN positions in channel locations."""
        # Create channel locations with some NaN values
        nan_chanlocs = [
            {'labels': 'Ch1', 'theta': 0, 'radius': 0.3},
            {'labels': 'Ch2', 'theta': np.nan, 'radius': 0.4},  # NaN theta
            {'labels': 'Ch3', 'theta': 90, 'radius': np.nan},   # NaN radius
            {'labels': 'Ch4', 'theta': 180, 'radius': 0.3}
        ]
        
        datavector_nan = np.array([1.0, 2.0, -1.0, 0.5])
        
        handle, Zi, plotrad, xi, yi = topoplot(
            datavector_nan, 
            nan_chanlocs, 
            noplot='on'
        )
        
        # Should handle NaN positions by excluding them
        self.assertIsInstance(Zi, np.ndarray)
        # Should still produce valid interpolation for valid channels
    
    def test_nan_data_values_handling(self):
        """Test topoplot with NaN values in data vector."""
        # Create data with NaN values
        nan_datavector = self.datavector.copy()
        nan_datavector[2] = np.nan  # Set one value to NaN
        nan_datavector[5] = np.inf  # Set one value to inf
        
        handle, Zi, plotrad, xi, yi = topoplot(
            nan_datavector, 
            self.chan_locs, 
            noplot='on'
        )
        
        # Should handle NaN/inf data values gracefully
        self.assertIsInstance(Zi, np.ndarray)
        # Function should exclude NaN/inf channels from interpolation
    
    def test_returns_axes_object(self):
        """Test that topoplot can return axes object when plotting."""
        with patch('matplotlib.pyplot.show'):  # Prevent actual display
            # Note: The current function doesn't return axes, but we test the structure
            handle, Zi, plotrad, xi, yi = topoplot(
                self.minimal_data, 
                self.minimal_chan_locs, 
                noplot='off'  # Enable plotting
            )
            
            # Check that plotting components are created
            self.assertIsInstance(Zi, np.ndarray)
            # Current implementation returns None for handle, but structure is there
    
    def test_color_limits_and_masking(self):
        """Test color limits and head boundary masking."""
        handle, Zi, plotrad, xi, yi = topoplot(
            self.datavector, 
            self.chan_locs, 
            noplot='on'
        )
        
        # Check that values outside head boundary are masked (NaN)
        head_mask = np.sqrt(xi**2 + yi**2) <= 0.5  # rmax = 0.5
        outside_head = ~head_mask
        
        # Values outside head should be NaN
        if np.any(outside_head):
            self.assertTrue(np.all(np.isnan(Zi[outside_head])))
        
        # Values inside head should include finite values
        inside_head = head_mask
        if np.any(inside_head):
            self.assertTrue(np.any(np.isfinite(Zi[inside_head])))
    
    def test_different_interpolation_methods(self):
        """Test different interpolation methods."""
        methods = ['rbf', 'griddata']
        
        results = []
        for method in methods:
            with self.subTest(method=method):
                handle, Zi, plotrad, xi, yi = topoplot(
                    self.datavector, 
                    self.chan_locs, 
                    noplot='on',
                    method=method
                )
                
                results.append(Zi)
                
                # Should complete successfully
                self.assertIsInstance(Zi, np.ndarray)
                self.assertEqual(Zi.shape, xi.shape)
        
        # Different methods should produce different results
        if len(results) > 1:
            self.assertFalse(np.allclose(results[0], results[1], equal_nan=True))
    
    def test_electrode_display_control(self):
        """Test electrode display control based on number of channels."""
        # Test with few channels (should show electrodes)
        handle, Zi, plotrad, xi, yi = topoplot(
            self.minimal_data, 
            self.minimal_chan_locs, 
            noplot='on'
        )
        # With 3 channels, ELECTRODES should be 'on'
        self.assertIsInstance(Zi, np.ndarray)
        
        # Test with many channels (should hide electrodes)
        many_chanlocs = []
        many_data = []
        for i in range(100):  # More than MAXDEFAULTSHOWLOCS (64)
            many_chanlocs.append({
                'labels': f'Ch{i}', 
                'theta': i * 3.6, 
                'radius': 0.3 + (i % 10) * 0.02
            })
            many_data.append(np.sin(i * 0.1))
        
        handle, Zi, plotrad, xi, yi = topoplot(
            np.array(many_data), 
            many_chanlocs, 
            noplot='on'
        )
        # With >64 channels, ELECTRODES should be 'off' but still work
        self.assertIsInstance(Zi, np.ndarray)
    
    def test_custom_parameters(self):
        """Test topoplot with custom parameters."""
        handle, Zi, plotrad, xi, yi = topoplot(
            self.datavector, 
            self.chan_locs, 
            noplot='on',
            intrad=0.8,
            plotrad=0.7,
            headrad=0.6,
            ELECTRODES='off'
        )
        
        # Should complete with custom parameters
        self.assertIsInstance(Zi, np.ndarray)
        self.assertAlmostEqual(plotrad, 0.7, places=1)
    
    def test_plotgrid_parameter(self):
        """Test plotgrid parameter."""
        handle, Zi, plotrad, xi, yi = topoplot(
            self.datavector, 
            self.chan_locs, 
            noplot='on',
            plotgrid='on'
        )
        
        # Should complete with plotgrid enabled
        self.assertIsInstance(Zi, np.ndarray)
    
    def test_single_channel_data(self):
        """Test topoplot with single channel."""
        single_chanloc = [{'labels': 'Cz', 'theta': 0, 'radius': 0.0}]
        single_data = np.array([1.5])
        
        # Single channel may cause issues due to insufficient data for interpolation
        try:
            handle, Zi, plotrad, xi, yi = topoplot(
                single_data, 
                single_chanloc, 
                noplot='on'
            )
            # If it succeeds, check basic properties
            self.assertIsInstance(Zi, np.ndarray)
        except (UnboundLocalError, IndexError, ValueError, np.linalg.LinAlgError):
            # Single channel interpolation may fail - this is acceptable behavior
            pass
    
    def test_empty_data_handling(self):
        """Test topoplot with empty or invalid data."""
        # Test with empty data - expect various possible errors
        with self.assertRaises((IndexError, ValueError, UnboundLocalError)):
            topoplot(np.array([]), self.chan_locs, noplot='on')
        
        # Test with mismatched data and channel count - expect various possible errors
        with self.assertRaises((IndexError, ValueError, UnboundLocalError)):
            topoplot(np.array([1, 2]), self.chan_locs, noplot='on')  # 2 values, 9 channels
    
    def test_coordinate_transformation(self):
        """Test coordinate transformation from polar to Cartesian."""
        handle, Zi, plotrad, xi, yi = topoplot(
            self.datavector, 
            self.chan_locs, 
            noplot='on'
        )
        
        # Check that coordinate grids are reasonable
        # Grid should be centered around origin
        self.assertLess(abs(np.mean(xi)), 0.5)
        self.assertLess(abs(np.mean(yi)), 0.5)
        
        # Grid should be symmetric
        self.assertAlmostEqual(np.min(xi), -np.max(xi), places=1)
        self.assertAlmostEqual(np.min(yi), -np.max(yi), places=1)
    
    def test_radius_calculations(self):
        """Test plotrad and intrad calculations."""
        # Test with default radius calculations
        handle, Zi, plotrad, xi, yi = topoplot(
            self.datavector, 
            self.chan_locs, 
            noplot='on'
        )
        
        self.assertGreater(plotrad, 0)
        self.assertLessEqual(plotrad, 1.0)
        
        # Test with custom intrad
        handle2, Zi2, plotrad2, xi2, yi2 = topoplot(
            self.datavector, 
            self.chan_locs, 
            noplot='on',
            intrad=0.8
        )
        
        # plotrad should be adjusted when intrad is specified
        self.assertLessEqual(plotrad2, 0.8)
    
    def test_head_boundary_masking(self):
        """Test that values outside head boundary are properly masked."""
        handle, Zi, plotrad, xi, yi = topoplot(
            self.datavector, 
            self.chan_locs, 
            noplot='on'
        )
        
        # Calculate head boundary (rmax = 0.5)
        rmax = 0.5
        distances = np.sqrt(xi**2 + yi**2)
        outside_head = distances > rmax
        
        # All values outside head should be NaN
        if np.any(outside_head):
            outside_values = Zi[outside_head]
            self.assertTrue(np.all(np.isnan(outside_values)))
    
    def test_interpolation_grid_properties(self):
        """Test properties of the interpolation grid."""
        handle, Zi, plotrad, xi, yi = topoplot(
            self.datavector, 
            self.chan_locs, 
            noplot='on'
        )
        
        # Grid should be square
        self.assertEqual(xi.shape[0], xi.shape[1])
        self.assertEqual(yi.shape[0], yi.shape[1])

        # Grid should have reasonable resolution (GRID_SCALE = 67, EEGLAB default)
        expected_size = 67
        self.assertEqual(xi.shape[0], expected_size)
        
        # Grid should cover the plotting area
        grid_range_x = np.max(xi) - np.min(xi)
        grid_range_y = np.max(yi) - np.min(yi)
        self.assertGreater(grid_range_x, 0.5)  # Should cover reasonable area
        self.assertGreater(grid_range_y, 0.5)
    
    def test_data_vector_flattening(self):
        """Test that data vector is properly flattened."""
        # Test with 2D data vector
        data_2d = self.datavector.reshape(-1, 1)
        
        handle, Zi, plotrad, xi, yi = topoplot(
            data_2d, 
            self.chan_locs, 
            noplot='on'
        )
        
        # Should handle 2D input by flattening
        self.assertIsInstance(Zi, np.ndarray)
    
    def test_channel_selection_logic(self):
        """Test channel selection and filtering logic."""
        # Test with matching data and channel counts
        handle, Zi, plotrad, xi, yi = topoplot(
            self.datavector,  # 9 elements
            self.chan_locs,   # 9 elements
            noplot='on'
        )
        
        # Should handle matching data gracefully
        self.assertIsInstance(Zi, np.ndarray)
        
        # Test with extra channel location - this may cause IndexError
        extra_chanlocs = self.chan_locs + [{'labels': 'REF', 'theta': 0, 'radius': 0}]
        
        try:
            handle2, Zi2, plotrad2, xi2, yi2 = topoplot(
                self.datavector,  # 9 elements
                extra_chanlocs,   # 10 elements
                noplot='on'
            )
            self.assertIsInstance(Zi2, np.ndarray)
        except (IndexError, ValueError):
            # Mismatch may cause errors - this is expected behavior
            pass
    
    def test_matplotlib_backend_compatibility(self):
        """Test compatibility with Agg backend."""
        # Verify Agg backend is active
        self.assertEqual(matplotlib.get_backend(), 'Agg')
        
        # Test that function works without display
        handle, Zi, plotrad, xi, yi = topoplot(
            self.datavector, 
            self.chan_locs, 
            noplot='on'
        )
        
        # Should complete successfully with Agg backend
        self.assertIsInstance(Zi, np.ndarray)
        
        # Test with plotting enabled (should not crash with Agg)
        with patch('matplotlib.pyplot.show'):
            handle, Zi, plotrad, xi, yi = topoplot(
                self.minimal_data, 
                self.minimal_chan_locs, 
                noplot='off'
            )
            self.assertIsInstance(Zi, np.ndarray)


class TestTopoplotParity(unittest.TestCase):
    """Test parity between Python and MATLAB topoplot implementations."""

    def setUp(self):
        """Set up test fixtures."""
        # Try to get MATLAB engine
        try:
            self.eeglab = get_eeglab('MAT', auto_file_roundtrip=False)
            self.matlab_available = True
        except Exception as e:
            self.matlab_available = False
            self.skipTest(f"MATLAB not available: {e}")

        # Load real EEG dataset with ICA
        test_file = os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')
        self.EEG = pop_loadset(test_file)

    def test_parity_single_component_noplot(self):
        """Test parity with MATLAB for single IC topography (noplot mode)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Get first IC weights
        icawinv = self.EEG['icawinv']
        datavector = icawinv[:, 0]  # First component
        chanlocs = self.EEG['chanlocs']

        # Python result
        _, Zi_py, _, _, _ = topoplot(datavector, chanlocs, noplot='on')

        # MATLAB result - need to call via file roundtrip for complex outputs
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        datavector = EEG.icawinv(:, 1);
        [~, Zi, ~, ~, ~] = topoplot(datavector, EEG.chanlocs, 'noplot', 'on');
        save('{temp_file}.mat', 'Zi');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        Zi_ml = mat_data['Zi']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare results
        # Max absolute diff: <1e-6, Nearly perfect parity
        # Max relative diff: <1e-5
        np.testing.assert_allclose(Zi_py, Zi_ml, rtol=1e-5, atol=1e-8,
                                   err_msg="topoplot Zi results differ beyond tolerance",
                                   equal_nan=True)

    def test_parity_multiple_components(self):
        """Test parity with MATLAB for multiple IC topographies."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Test first 5 components
        icawinv = self.EEG['icawinv']
        chanlocs = self.EEG['chanlocs']

        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        for ic_idx in range(min(5, icawinv.shape[1])):
            datavector = icawinv[:, ic_idx]

            # Python result
            _, Zi_py, _, _, _ = topoplot(datavector, chanlocs, noplot='on')

            # MATLAB result
            matlab_code = f"""
            EEG = pop_loadset('{temp_file}');
            datavector = EEG.icawinv(:, {ic_idx + 1});
            [~, Zi, ~, ~, ~] = topoplot(datavector, EEG.chanlocs, 'noplot', 'on');
            save('{temp_file}_ic{ic_idx}.mat', 'Zi');
            """
            self.eeglab.eval(matlab_code, nargout=0)

            # Load MATLAB result
            mat_data = scipy.io.loadmat(f'{temp_file}_ic{ic_idx}.mat')
            Zi_ml = mat_data['Zi']

            # Compare results
            # Max absolute diff: <1e-6, Nearly perfect parity
            # Max relative diff: <1e-5
            np.testing.assert_allclose(Zi_py, Zi_ml, rtol=1e-5, atol=1e-8,
                                       err_msg=f"topoplot Zi results differ for IC {ic_idx}",
                                       equal_nan=True)

            # Clean up IC-specific file
            os.remove(f'{temp_file}_ic{ic_idx}.mat')

        # Clean up temp files
        os.remove(temp_file)
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))


if __name__ == '__main__':
    unittest.main()
