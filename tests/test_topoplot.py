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
from matplotlib.collections import PathCollection
from unittest.mock import patch
import tempfile
import warnings
import scipy.io
import pytest

# Set Agg backend before importing topoplot to avoid display issues
matplotlib.use('Agg')

from eegprep.functions.sigprocfunc.topoplot import topoplot, griddata_v4, topo_screen_coords
from eegprep import pop_loadset, pop_saveset
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from tests.eeglab_tests import eeglab_test

local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')


@pytest.mark.gui
@eeglab_test("unittesting_sigprocfunc/topoplot/sigprocfunc_topoplot_wrapperTest.m", "test_test_topoplot")
def test_reference_topoplot(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    indices = np.asarray(eeg["icachansind"]).ravel(order="F").astype(int) - 1
    eeglab_backend(
        "topoplot",
        eeg["icawinv"][:, :1],
        np.asarray(eeg["chanlocs"])[..., indices],
        verbose="off",
        electrodes="on",
        style="both",
        plotrad=0.55,
        intrad=0.55,
        noplot="on",
        chaninfo=eeg["chaninfo"],
        nargout=5,
    )


class TestGriddataV4(unittest.TestCase):
    """Test the griddata_v4 function (biharmonic spline interpolation)."""

    def test_interpolation_does_not_leak_finite_matmul_warnings(self):
        theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        values = np.linspace(-1, 1, 32)
        query = np.linspace(-1, 1, 67)
        xq, yq = np.meshgrid(query, query)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            interpolated = griddata_v4(x, y, values, xq, yq)

        assert np.isfinite(interpolated).all()


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
            {'labels': 'P4', 'theta': 135, 'radius': 0.5},
        ]

        # Create synthetic data vector (one value per channel)
        self.datavector = np.array([1.0, 2.0, -1.0, 0.5, -0.5, 1.5, -1.5, 0.8, -0.8])

        # Create minimal channel locations for edge cases
        self.minimal_chan_locs = [
            {'labels': 'Cz', 'theta': 0, 'radius': 0.0},
            {'labels': 'Fz', 'theta': 0, 'radius': 0.3},
            {'labels': 'Pz', 'theta': 180, 'radius': 0.3},
        ]
        self.minimal_data = np.array([1.0, 0.5, -0.5])

    def test_head_boundary_masking(self):
        """Test that values outside head boundary are properly masked."""
        handle, Zi, plotrad, xi, yi = topoplot(self.datavector, self.chan_locs, noplot='on')

        # Calculate head boundary (rmax = 0.5)
        rmax = 0.5
        distances = np.sqrt(xi**2 + yi**2)
        outside_head = distances > rmax

        # All values outside head should be NaN
        if np.any(outside_head):
            outside_values = Zi[outside_head]
            self.assertTrue(np.all(np.isnan(outside_values)))

    def test_topo_screen_coords_cardinal_directions(self):
        """Pin the shared polar-to-screen orientation contract (nose up, EEGLAB left-right).

        theta=0 -> front (+y), 90 -> right (+x), 180 -> back (-y), 270 -> left (-x).
        Three modules import this helper, so a sign flip here would re-mirror every plot.
        """
        cardinals = {
            0: (0.0, 0.5),  # front (nose up)
            90: (0.5, 0.0),  # right
            180: (0.0, -0.5),  # back
            270: (-0.5, 0.0),  # left
        }
        for theta_deg, (expected_x, expected_y) in cardinals.items():
            screen_x, screen_y = topo_screen_coords(theta_deg, 0.5)
            self.assertAlmostEqual(float(screen_x), expected_x, places=10)
            self.assertAlmostEqual(float(screen_y), expected_y, places=10)

    def test_markers_match_eeglab_left_right_orientation(self):
        """Markers sit on the same side as their data, matching EEGLAB (no L/R mirror).

        EEGLAB plots screen X = sin(theta)*Rd, so a theta=90 channel is on the right.
        Its marker and its interpolated data peak must land on the same (right) side.
        """
        chan_locs = [
            {'labels': 'FRONT', 'theta': 0, 'radius': 0.5},
            {'labels': 'RIGHT', 'theta': 90, 'radius': 0.5},
            {'labels': 'BACK', 'theta': 180, 'radius': 0.5},
            {'labels': 'LEFT', 'theta': 270, 'radius': 0.5},
        ]
        data = np.array([0.0, 10.0, 0.0, 0.0])  # hot spot on the theta=90 (EEGLAB right) channel
        fig, ax = plt.subplots()
        try:
            with patch('matplotlib.pyplot.show'):
                topoplot(data, chan_locs, axes=ax, electrodes='on')
            markers = next(
                np.asarray(c.get_offsets())
                for c in ax.collections
                if isinstance(c, PathCollection) and len(c.get_offsets()) == len(chan_locs)
            )
            # Marker order matches chan_locs: FRONT, RIGHT, BACK, LEFT.
            self.assertGreater(markers[1, 0], 0)  # RIGHT marker on the right (screen_x > 0)
            self.assertLess(markers[3, 0], 0)  # LEFT marker on the left (screen_x < 0)
            self.assertGreater(markers[0, 1], markers[2, 1])  # FRONT above BACK (screen_y)
            image = ax.images[0]
            grid = image.get_array()
            left, right, _, _ = image.get_extent()
            _, col = np.unravel_index(np.nanargmax(grid), grid.shape)
            blob_x = left + (col + 0.5) / grid.shape[1] * (right - left)
            self.assertGreater(blob_x, 0)  # data blob on the same (right) side as the marker
        finally:
            plt.close(fig)

    def test_extent_matches_channel_geometry_for_asymmetric_layout(self):
        """Image extent uses (xmin, xmax) for front-back, not the old (-xmax, -xmin).

        A back channel beyond the head (radius > 1) makes xmin != -xmax; the image
        bottom must follow xmin so the interpolated field aligns with channel geometry.
        The old (-xmax, -xmin) form would clamp the bottom to -rmax instead.
        """
        chan_locs = [
            {'labels': 'BACK', 'theta': 180, 'radius': 1.2},  # off-head -> asymmetric xmin
            {'labels': 'RIGHT', 'theta': 90, 'radius': 0.5},
            {'labels': 'FRONT', 'theta': 0, 'radius': 0.5},
        ]
        data = np.array([1.0, 0.0, -1.0])
        fig, ax = plt.subplots()
        try:
            with patch('matplotlib.pyplot.show'):
                topoplot(data, chan_locs, axes=ax, electrodes='on')
            _, _, bottom, top = ax.images[0].get_extent()
            self.assertLess(bottom, -0.5)  # extends below -rmax; old (-xmax,-xmin) would give -0.5
            self.assertAlmostEqual(top, 0.5, places=6)  # front side stays at +rmax
        finally:
            plt.close(fig)

    def test_labelpoint_offsets_labels_from_dots(self):
        """labelpoint labels sit beside the marker, not on top (issue #299)."""
        with patch('matplotlib.pyplot.show'):
            fig, _, _, _, _ = topoplot(None, self.chan_locs, style='blank', electrodes='labelpoint')
        try:
            ax = fig.axes[0]
            annotations = {a.get_text(): a for a in ax.texts}
            self.assertIn('Fz', annotations)
            offsets = []
            for loc in self.chan_locs:
                label = loc['labels']
                theta = np.deg2rad(loc['theta'])
                r = loc['radius']
                dot_x = np.sin(theta) * r
                text_x, _ = annotations[label].get_position()
                offsets.append(text_x - dot_x)
                self.assertEqual(annotations[label].get_ha(), 'left')
            # All labels shifted by the same positive offset from their dot.
            self.assertGreater(offsets[0], 0)
            for offset in offsets[1:]:
                self.assertAlmostEqual(offset, offsets[0], places=6)
        finally:
            plt.close(fig)

        with patch('matplotlib.pyplot.show'):
            fig, _, _, _, _ = topoplot(None, self.chan_locs, style='blank', electrodes='labels')
        try:
            ax = fig.axes[0]
            fz = next(a for a in ax.texts if a.get_text() == 'Fz')
            theta = np.deg2rad(self.chan_locs[0]['theta'])
            r = self.chan_locs[0]['radius']
            self.assertAlmostEqual(fz.get_position()[0], np.sin(theta) * r, places=6)
            self.assertEqual(fz.get_ha(), 'center')
        finally:
            plt.close(fig)

    def test_showlabels_with_electrodes_on_offsets_labels(self):
        """electrodes='on' with showlabels sits labels beside the dot, not on it (issue #299)."""
        with patch('matplotlib.pyplot.show'):
            handle, _, _, _, _ = topoplot(self.datavector, self.chan_locs, electrodes='on', showlabels=True)
        try:
            labels_drawn = [a for a in handle.axes[0].texts if a.get_text()]
            self.assertTrue(labels_drawn)
            self.assertTrue(all(a.get_ha() == 'left' for a in labels_drawn))
        finally:
            plt.close(handle)


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
        np.testing.assert_allclose(
            Zi_py, Zi_ml, rtol=1e-5, atol=1e-8, err_msg="topoplot Zi results differ beyond tolerance", equal_nan=True
        )

    def test_parity_gridscale_32(self):
        """Test parity with MATLAB for gridscale=32 (ICL_feature_extractor usage)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Get first IC weights
        icawinv = self.EEG['icawinv']
        datavector = icawinv[:, 0]  # First component
        chanlocs = self.EEG['chanlocs']

        # Python result with gridscale=32
        _, Zi_py, _, _, _ = topoplot(datavector, chanlocs, noplot='on', gridscale=32)

        # MATLAB result - need to specify gridscale
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        datavector = EEG.icawinv(:, 1);
        [~, Zi, ~, ~, ~] = topoplot(datavector, EEG.chanlocs, 'noplot', 'on', 'gridscale', 32);
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
        # Max absolute diff: TBD
        # Max relative diff: TBD
        np.testing.assert_allclose(
            Zi_py, Zi_ml, rtol=1e-5, atol=1e-8, err_msg="topoplot Zi results differ for gridscale=32", equal_nan=True
        )


if __name__ == '__main__':
    unittest.main()
