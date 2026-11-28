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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.markers import MarkerStyle
from unittest.mock import patch
import tempfile
import warnings
import scipy.io
import pytest

# Set Agg backend before importing topoplot to avoid display issues
matplotlib.use('Agg')

from eegprep.functions.sigprocfunc.topoplot import (
    _ALLOWED_OPTIONS,
    _UNIMPLEMENTED_EEGLAB_OPTIONS,
    _average_similar_points,
    _merge_coincident_points,
    griddata_v4,
    topo_screen_coords,
    topoplot,
)
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

    def test_coincident_points_average_like_matlab_griddata(self):
        theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        x, y = np.cos(theta), np.sin(theta)
        values = np.linspace(-1, 1, 8)
        query = np.linspace(-1, 1, 11)
        xq, yq = np.meshgrid(query, query)

        merged = values.copy()
        merged[0] = 0.5 * (values[0] + 3.0)
        expected = griddata_v4(x, y, merged, xq, yq)
        with self.assertWarns(UserWarning) as caught:
            actual = griddata_v4(np.append(x, x[0]), np.append(y, y[0]), np.append(values, 3.0), xq, yq)

        self.assertIn(_DUPLICATE_POINT_WARNING, str(caught.warning))
        np.testing.assert_allclose(actual, expected)

    def test_exact_tolerance_boundary_matches_mergesimpts(self):
        # mergesimpts([0; 0.5; 1], 0.5, 'average') keeps all three because the
        # comparison is strict <. nextafter on either side of 0.5 decides the merge.
        y = np.zeros(3)
        values = np.array([1.0, 2.0, 3.0])
        kept = _average_similar_points(np.array([0.0, 0.5, 1.0]), y, values, 0.5, np.inf)
        self.assertEqual(kept[0].size, 3)
        np.testing.assert_allclose(kept[0], [0.0, 0.5, 1.0])

        inside = np.nextafter(0.5, -np.inf)
        merged = _average_similar_points(np.array([0.0, inside, 1.0]), y, values, 0.5, np.inf)
        self.assertEqual(merged[0].size, 2)
        np.testing.assert_allclose(merged[0], [inside / 2.0, 1.0])

        # The far point stays more than tol from both neighbors. Putting it at 1
        # would make |1 - nextafter(0.5, +inf)| itself smaller than 0.5.
        outside = np.nextafter(0.5, np.inf)
        separate = _average_similar_points(np.array([0.0, outside, 2.0]), y, values, 0.5, np.inf)
        self.assertEqual(separate[0].size, 3)
        np.testing.assert_allclose(separate[0], [0.0, outside, 2.0])

        corners = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 2.0], [1.0, 1.0, 3.0]])
        anchor = np.array([[0.0, 0.0, 4.0]])
        span = 2.0
        tol = float(np.spacing(0.5 * span) ** (1.0 / 3.0))
        for offset, expected_count in (
            (tol, 6),
            (np.nextafter(tol, -np.inf), 5),
            (np.nextafter(tol, np.inf), 6),
        ):
            with self.subTest(offset=offset, expected_count=expected_count):
                extra = np.array([[offset, offset, 5.0]])
                points = np.vstack((corners, anchor, extra))
                merged_x, merged_y, _merged_v, merged_count = _merge_coincident_points(
                    points[:, 0], points[:, 1], points[:, 2]
                )
                self.assertEqual(merged_x.size, expected_count)
                self.assertEqual(merged_y.size, expected_count)
                self.assertEqual(merged_count > 0, expected_count < points.shape[0])

    def test_one_unique_location_names_the_two_sensor_requirement(self):
        query = np.linspace(-1, 1, 5)
        xq, yq = np.meshgrid(query, query)
        with self.assertWarns(UserWarning) as caught:
            with self.assertRaisesRegex(ValueError, _TWO_LOCATIONS_ERROR):
                griddata_v4(
                    np.array([0.2, 0.2, 0.2]),
                    np.array([-0.1, -0.1, -0.1]),
                    np.array([1.0, 3.0, 5.0]),
                    xq,
                    yq,
                )
        self.assertIn(_DUPLICATE_POINT_WARNING, str(caught.warning))
        with warnings.catch_warnings(record=True) as caught_single:
            warnings.simplefilter("always", UserWarning)
            with self.assertRaisesRegex(ValueError, _TWO_LOCATIONS_ERROR):
                griddata_v4(np.array([0.2]), np.array([-0.1]), np.array([1.0]), xq, yq)
        self.assertFalse(any("griddata_v4" in str(item.message) for item in caught_single))

    def test_near_duplicate_points_merge_to_the_averaged_sensor(self):
        x, y, values, xq, yq = _circle_samples()
        for offset in (1e-9, 1e-7):
            with self.subTest(offset=offset):
                expected = _averaged_duplicate_map(x, y, values, xq, yq, offset)
                with self.assertWarns(UserWarning) as caught:
                    actual = griddata_v4(
                        np.append(x, x[0] + offset),
                        np.append(y, y[0]),
                        np.append(values, 3.0),
                        xq,
                        yq,
                    )
                self.assertIn(_DUPLICATE_POINT_WARNING, str(caught.warning))
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_points_one_millimeter_apart_on_the_unit_circle_stay_separate(self):
        x, y, values, xq, yq = _circle_samples()
        offset = 1e-3
        collapsed = _averaged_duplicate_map(x, y, values, xq, yq, offset)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            actual = griddata_v4(
                np.append(x, x[0] + offset),
                np.append(y, y[0]),
                np.append(values, 3.0),
                xq,
                yq,
            )
        self.assertFalse(any("griddata_v4" in str(item.message) for item in caught))
        self.assertFalse(np.allclose(actual, collapsed, atol=1e-8, rtol=1e-6))

    def test_chain_merges_only_points_within_tolerance_of_the_anchor(self):
        # Corners fix the axis span so the tolerance stays near 6e-6. A chain
        # A-B-C with each step inside that tolerance still leaves C out when C
        # is outside tolerance of A. A point that sorts between A and D does
        # not stop D joining A when D itself is within tolerance of A.
        corners = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.5], [1.0, 1.0, 1.0]])
        query = np.linspace(-1, 1, 11)
        xq, yq = np.meshgrid(query, query)
        chains = {
            "stops_at_anchor": (
                np.array([[0.0, 0.0, 1.0], [4e-6, 0.0, 3.0], [9e-6, 0.0, 7.0]]),
                np.array([[2e-6, 0.0, 2.0], [9e-6, 0.0, 7.0]]),
            ),
            "keeps_point_past_an_outlier": (
                np.array([[0.0, 0.0, 1.0], [0.5, 1e-6, 10.0], [1e-6, 2e-6, 3.0]]),
                np.array([[5e-7, 1e-6, 2.0], [0.5, 1e-6, 10.0]]),
            ),
        }
        for name, (cluster, merged_cluster) in chains.items():
            with self.subTest(chain=name):
                points = np.vstack((corners, cluster))
                merged = np.vstack((corners, merged_cluster))
                with self.assertWarns(UserWarning) as caught:
                    actual = griddata_v4(points[:, 0], points[:, 1], points[:, 2], xq, yq)
                self.assertIn(_DUPLICATE_POINT_WARNING, str(caught.warning))
                expected = griddata_v4(merged[:, 0], merged[:, 1], merged[:, 2], xq, yq)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


_DUPLICATE_POINT_WARNING = (
    "topoplot griddata_v4: Duplicate x-y data points detected: using average values for duplicate points."
)
_TWO_LOCATIONS_ERROR = (
    r"topoplot griddata_v4: at least two distinct sensor locations are needed\. "
    r"Pass values at two or more electrodes that stay apart after duplicate averaging\."
)


def _circle_samples():
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    x, y = np.cos(theta), np.sin(theta)
    values = np.linspace(-1, 1, 8)
    query = np.linspace(-1, 1, 11)
    xq, yq = np.meshgrid(query, query)
    return x, y, values, xq, yq


def _averaged_duplicate_map(x, y, values, xq, yq, offset):
    merged_x = x.copy()
    merged_values = values.copy()
    merged_x[0] = x[0] + offset / 2
    merged_values[0] = 0.5 * (values[0] + 3.0)
    return griddata_v4(merged_x, y, merged_values, xq, yq)


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

        self.minimal_chan_locs = [
            {'labels': 'Cz', 'theta': 0, 'radius': 0.0},
            {'labels': 'Fz', 'theta': 0, 'radius': 0.3},
            {'labels': 'Pz', 'theta': 180, 'radius': 0.3},
        ]
        self.minimal_data = np.array([1.0, 0.5, -0.5])

    def test_head_boundary_masking(self):
        """Test that values outside head boundary are properly masked."""
        handle, Zi, plotrad, xi, yi = topoplot(self.datavector, self.chan_locs, noplot='on')

        rmax = 0.5
        distances = np.sqrt(xi**2 + yi**2)
        outside_head = distances > rmax

        # All values outside head should be NaN
        if np.any(outside_head):
            outside_values = Zi[outside_head]
            self.assertTrue(np.all(np.isnan(outside_values)))

    def test_nonfinite_value_drops_its_channel_without_shifting_others(self):
        data = self.datavector.copy()
        data[2] = np.nan
        keep = [i for i in range(len(data)) if i != 2]

        _, actual, _, _, _ = topoplot(data, self.chan_locs, noplot='on')
        _, expected, _, _, _ = topoplot(data[keep], [self.chan_locs[i] for i in keep], noplot='on')

        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_nonfinite_value_at_either_end_or_the_middle_matches_dropping_that_channel(self):
        for bad in (np.nan, np.inf, -np.inf):
            for index in (0, len(self.datavector) // 2, len(self.datavector) - 1):
                with self.subTest(bad=bad, index=index):
                    data = self.datavector.copy()
                    data[index] = bad
                    keep = [i for i in range(len(data)) if i != index]
                    _, actual, _, _, _ = topoplot(data, self.chan_locs, noplot="on")
                    _, expected, _, _, _ = topoplot(data[keep], [self.chan_locs[i] for i in keep], noplot="on")
                    np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_one_extra_channel_location_drops_the_common_reference(self):
        locs = [
            {"labels": f"E{index}", "theta": index * (360.0 / 16.0), "radius": 0.4 if index else 0.0}
            for index in range(16)
        ]
        locs.append({"labels": "Ref", "theta": 10.0, "radius": 0.2})
        data = np.linspace(-1.0, 1.0, 16)
        _, actual, _, _, _ = topoplot(data, locs, noplot="on")
        _, expected, _, _, _ = topoplot(data, locs[:-1], noplot="on")
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_single_plot_channel_asks_for_two_locations(self):
        for index in (1, 5):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, _TWO_LOCATIONS_ERROR):
                    topoplot(self.datavector, self.chan_locs, noplot="on", plotchans=[index])
        with self.assertRaisesRegex(ValueError, _TWO_LOCATIONS_ERROR):
            topoplot(np.array([self.datavector[0]]), [self.chan_locs[0]], noplot="on")

    def test_plotchans_keeps_the_requested_channels(self):
        keep = [1, 6, 9]
        selected = [index - 1 for index in keep]
        _, actual, _, _, _ = topoplot(self.datavector, self.chan_locs, noplot="on", plotchans=keep)
        _, expected, _, _, _ = topoplot(
            self.datavector[selected],
            [self.chan_locs[index] for index in selected],
            noplot="on",
        )
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_plotchans_includes_the_last_channel(self):
        last = len(self.chan_locs)
        selected = [0, last - 1]
        _, actual, _, _, _ = topoplot(self.datavector, self.chan_locs, noplot="on", plotchans=[1, last])
        _, expected, _, _, _ = topoplot(
            self.datavector[selected],
            [self.chan_locs[index] for index in selected],
            noplot="on",
        )
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_plotchans_outside_the_channel_list_names_the_fix(self):
        message = (
            r"topoplot: plotchans must be 1-based indices in 1\.\.9\. "
            r"Got \[0, 99\]\. Pass the channel indices to include, for example \[1, 3\]\."
        )
        with self.assertRaisesRegex(ValueError, message):
            topoplot(self.datavector, self.chan_locs, noplot="on", plotchans=[0, 99])

    def test_missing_or_empty_channel_locations_name_the_problem(self):
        message = (
            r"topoplot: Channel locations are required\. "
            r"Load locations with pop_chanedit or pop_readlocs before plotting\."
        )
        with self.assertRaisesRegex(ValueError, message):
            topoplot(np.arange(4.0), None, noplot="on")
        with self.assertRaisesRegex(ValueError, message):
            topoplot(np.arange(4.0), [], noplot="on")
        with self.assertRaisesRegex(ValueError, message):
            topoplot(np.arange(4.0), np.array([]), noplot="on")
        with self.assertRaisesRegex(ValueError, message):
            topoplot([], None, style="blank", noplot="on")

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

        test_file = os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')
        self.EEG = pop_loadset(test_file)

    def test_parity_single_component_noplot(self):
        """Test parity with MATLAB for single IC topography (noplot mode)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

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


def _matlab_chanlocs(locations):
    # readlocs keeps a channel only when both theta and X are non-empty.
    records = np.zeros(
        len(locations),
        dtype=[
            ("labels", "U16"),
            ("theta", float),
            ("radius", float),
            ("X", float),
            ("Y", float),
            ("Z", float),
        ],
    )
    for index, location in enumerate(locations):
        theta = np.deg2rad(location["theta"])
        radius = float(location["radius"])
        records[index] = (
            location["labels"],
            location["theta"],
            radius,
            np.cos(theta) * radius,
            np.sin(theta) * radius,
            0.0,
        )
    return records


@pytest.mark.matlab_oracle
@pytest.mark.parity
def test_reference_near_duplicate_and_nan_channel(eeglab_backend):
    """Compare near-duplicate and NaN maps with MATLAB topoplot under the existing Zi tolerance."""
    theta = np.linspace(0, 360, 8, endpoint=False)
    locations = [
        {"labels": f"E{index}", "theta": float(angle), "radius": 0.4 if index else 0.0}
        for index, angle in enumerate(theta)
    ]
    values = np.linspace(-1.0, 1.0, len(locations))
    near = [dict(location) for location in locations]
    near.append({"labels": "Dup", "theta": locations[1]["theta"], "radius": locations[1]["radius"] + 1e-9})
    near_values = np.append(values, 3.0)
    missing = values.copy()
    missing[0] = np.nan

    def _zi(data, locs):
        _handle, zi, _plotrad, _xi, _yi = topoplot(data, locs, noplot="on", verbose="off")
        return zi

    def _matlab_zi(data, locs):
        outputs = eeglab_backend(
            "topoplot",
            np.asarray(data, dtype=float).reshape(-1, 1),
            _matlab_chanlocs(locs),
            noplot="on",
            verbose="off",
            nargout=5,
        )
        return np.asarray(outputs[1], dtype=float)

    np.testing.assert_allclose(
        _zi(near_values, near),
        _matlab_zi(near_values, near),
        rtol=1e-5,
        atol=1e-8,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        _zi(missing, locations),
        _matlab_zi(missing, locations),
        rtol=1e-5,
        atol=1e-8,
        equal_nan=True,
    )


def test_topoplot_rejects_unknown_options_and_keeps_compat_allowlist():
    locs = [
        {"labels": "Fz", "theta": 0.0, "radius": 0.3},
        {"labels": "Cz", "theta": 0.0, "radius": 0.0},
        {"labels": "Pz", "theta": 180.0, "radius": 0.3},
        {"labels": "C3", "theta": 270.0, "radius": 0.4},
    ]
    data = np.array([1.0, 2.0, -1.0, 0.5])

    assert _ALLOWED_OPTIONS.isdisjoint(_UNIMPLEMENTED_EEGLAB_OPTIONS)

    # Truly unknown parameters raise 'Unknown input parameter'.
    for bad_key, bad_val in (("foobar", 1), ("not_a_topoplot_opt", "on")):
        with pytest.raises(ValueError, match=rf"topoplot: Unknown input parameter '{bad_key}'"):
            topoplot(data, locs, noplot="on", **{bad_key: bad_val})
        with pytest.raises(ValueError, match=rf"topoplot: Unknown input parameter '{bad_key}'"):
            topoplot([], locs, style="blank", noplot="on", **{bad_key: bad_val})

    # Valid EEGLAB topoplot.m options not yet implemented raise an actionable message.
    for unimpl_key in sorted(_UNIMPLEMENTED_EEGLAB_OPTIONS):
        pattern = rf"topoplot: '{unimpl_key}' is an EEGLAB topoplot option that EEGPrep does not implement yet\. Remove it from the call\."
        with pytest.raises(ValueError, match=pattern):
            topoplot(data, locs, noplot="on", **{unimpl_key: 1})
    for unimpl_key in ("headrad", "plotgrid", "dipole", "whitebk", "shading", "nosedir"):
        pattern = rf"topoplot: '{unimpl_key}' is an EEGLAB topoplot option that EEGPrep does not implement yet\. Remove it from the call\."
        with pytest.raises(ValueError, match=pattern):
            topoplot([], locs, style="blank", noplot="on", **{unimpl_key: 1})

    # Explicitly ignored / caller-compat keys remain accepted.
    _handle, zi, *_rest = topoplot(
        data,
        locs,
        noplot="on",
        iclabel=True,
        verbose="off",
        chaninfo={},
        masksurf="on",
    )
    assert np.isfinite(zi).any()


def test_topoplot_emarker_draws_marker_color_size_linewidth_and_autosize():
    locs = [
        {"labels": "Fz", "theta": 0.0, "radius": 0.3},
        {"labels": "Cz", "theta": 0.0, "radius": 0.0},
        {"labels": "Pz", "theta": 180.0, "radius": 0.3},
        {"labels": "C3", "theta": 270.0, "radius": 0.4},
        {"labels": "C4", "theta": 90.0, "radius": 0.4},
    ]
    data = np.array([1.0, 2.0, -1.0, 0.5, -0.5])

    # Full 4-item emarker tuple draws hollow blue squares with explicit size and linewidth.
    fig, ax = plt.subplots()
    try:
        topoplot(data, locs, axes=ax, emarker=["s", [0.0, 0.0, 1.0], 8, 2])
        marker_col = next(c for c in ax.collections if isinstance(c, PathCollection))
        assert len(marker_col.get_offsets()) == 5
        assert np.asarray(marker_col.get_facecolor()).size == 0
        np.testing.assert_allclose(marker_col.get_edgecolor()[0], [0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(marker_col.get_sizes(), [64.0])
        np.testing.assert_allclose(marker_col.get_linewidth(), [2.0])
    finally:
        plt.close(fig)

    # Empty size placeholder triggers EEGLAB channel-count auto-size table.
    # For '.', size_sq = (2 * pt / 3) ** 2 and linewidths == 0 to match MATLAB MarkerSize / 3.
    for n_chans, expected_pt in ((8, 10.0), (32, 8.0), (48, 6.0), (64, 5.0), (80, 4.0), (100, 3.0)):
        ring_locs = [
            {"labels": f"E{i + 1}", "theta": i * (360.0 / n_chans), "radius": 0.4 if i else 0.0} for i in range(n_chans)
        ]
        ring_data = np.linspace(-1.0, 1.0, n_chans)
        fig, ax = plt.subplots()
        try:
            topoplot(ring_data, ring_locs, axes=ax, electrodes="on", emarker=[".", "r", [], 1.5])
            marker_col = next(c for c in ax.collections if isinstance(c, PathCollection))
            assert len(marker_col.get_offsets()) == n_chans
            np.testing.assert_allclose(marker_col.get_sizes(), [(2.0 * expected_pt / 3.0) ** 2])
            np.testing.assert_allclose(marker_col.get_linewidth(), [0.0])
            np.testing.assert_allclose(marker_col.get_facecolor()[0], [1.0, 0.0, 0.0, 1.0])
        finally:
            plt.close(fig)

    # Legacy markersize= kwarg stays backward-compatible when no emarker size / emsize is given.
    fig, ax = plt.subplots()
    try:
        topoplot(data, locs, axes=ax, markersize=25)
        marker_col = next(c for c in ax.collections if isinstance(c, PathCollection))
        np.testing.assert_allclose(marker_col.get_sizes(), [25.0])
    finally:
        plt.close(fig)

    # Left-to-right option precedence between emarker and emsize/ecolor.
    fig, ax = plt.subplots()
    try:
        topoplot(data, locs, axes=ax, emarker=["o", "r", 12, 1], emsize=6, ecolor="b")
        marker_col = next(c for c in ax.collections if isinstance(c, PathCollection))
        np.testing.assert_allclose(marker_col.get_sizes(), [36.0])
        np.testing.assert_allclose(marker_col.get_edgecolor()[0], [0.0, 0.0, 1.0, 1.0])
    finally:
        plt.close(fig)

    # Unfilled line markers ('+', 'x', '*') draw with zero matplotlib warnings; 'none' skips drawing.
    for sym in ("+", "x", "*"):
        fig, ax = plt.subplots()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                topoplot(data, locs, axes=ax, emarker=[sym, "k", 9, 1.5])
                fig.canvas.draw()
            assert not any("unfilled marker" in str(item.message) for item in caught)
            marker_col = next(c for c in ax.collections if isinstance(c, PathCollection))
            assert len(marker_col.get_offsets()) == 5
        finally:
            plt.close(fig)

    fig, ax = plt.subplots()
    try:
        topoplot(data, locs, axes=ax, emarker="none")
        assert not any(isinstance(c, PathCollection) for c in ax.collections)
    finally:
        plt.close(fig)

    for bad_emarker in ([], [".", "k", 8, 1, "extra"], "not_a_marker", [".", "not_a_color"]):
        with pytest.raises(ValueError, match="topoplot"):
            topoplot(data, locs, noplot="on", emarker=bad_emarker)


def test_topoplot_dot_marker_diameter_matches_matlab_one_third_rule():
    """Rendered '.' dot diameter at 600 DPI matches MATLAB's MarkerSize / 3 rule (~3.24 pt at 10, ~6.60 pt at 20)."""
    locs = [{"labels": "Cz", "theta": 0.0, "radius": 0.0}]
    dpi = 600

    def _measured_dot_diameter_pt(emarkersize: float, linewidth: float = 1.0) -> tuple[float, float]:
        fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=dpi)
        try:
            fig.patch.set_facecolor("white")
            topoplot(
                1,
                locs,
                style="blank",
                electrodes="off",
                emarker=[".", "k", emarkersize, linewidth],
                emarkersize1chan=emarkersize,
                axes=ax,
                title="",
            )
            marker_col = next(c for c in ax.collections if isinstance(c, PathCollection))
            np.testing.assert_allclose(marker_col.get_linewidth(), [0.0])
            canvas = fig.canvas
            assert isinstance(canvas, FigureCanvasAgg)
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            # Selected dot is red [255, 0, 0] at Cz (center); isolate red pixels cleanly from black head outline.
            red_mask = (rgba[:, :, 0] > 180) & (rgba[:, :, 1] < 80) & (rgba[:, :, 2] < 80)
            ys, xs = np.nonzero(red_mask)
            assert xs.size > 0 and ys.size > 0
            width_pt = (int(xs.max()) - int(xs.min()) + 1) * 72.0 / dpi
            height_pt = (int(ys.max()) - int(ys.min()) + 1) * 72.0 / dpi
            return width_pt, height_pt
        finally:
            plt.close(fig)

    w10, h10 = _measured_dot_diameter_pt(10.0, linewidth=1.0)
    w10_thick, h10_thick = _measured_dot_diameter_pt(10.0, linewidth=5.0)
    w20, h20 = _measured_dot_diameter_pt(20.0, linewidth=1.0)

    # MATLAB MarkerSize / 3: 10 pt -> ~3.24 pt (not ~5.88 pt), 20 pt -> ~6.60 pt (not ~10.88 pt).
    assert 2.9 <= w10 <= 3.6 and 2.9 <= h10 <= 3.6
    assert 6.1 <= w20 <= 7.1 and 6.1 <= h20 <= 7.1
    # LineWidth does not inflate '.' dot diameter, matching MATLAB.
    assert w10_thick == pytest.approx(w10, abs=0.15)
    assert h10_thick == pytest.approx(h10, abs=0.15)


def test_topoplot_matlab_star_and_asterisk_marker_mappings():
    """'*' draws an unfilled 8-spoke asterisk; 'p'/'pentagram' and 'h'/'hexagram' draw fillable stars."""
    locs = [
        {"labels": "Fz", "theta": 0.0, "radius": 0.3},
        {"labels": "Cz", "theta": 0.0, "radius": 0.0},
        {"labels": "Pz", "theta": 180.0, "radius": 0.3},
        {"labels": "C3", "theta": 270.0, "radius": 0.4},
    ]
    data = np.array([1.0, 2.0, -1.0, 0.5])

    # MATLAB '*' -> 8-spoke unfilled line asterisk in both emarker and emarker2 with zero warnings.
    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            topoplot(data, locs, axes=ax, emarker=["*", "b", 10, 1.5], emarker2=[[1], "*", "r", 12, 2.0])
            fig.canvas.draw()
        assert not any("unfilled marker" in str(item.message) for item in caught)
        pcs = sorted((c for c in ax.collections if isinstance(c, PathCollection)), key=lambda c: c.get_zorder())
        assert len(pcs) == 2
        # Unfilled 8-spoke asterisk matches Matplotlib's transformed (8, 2, 0) path.
        asterisk_style = MarkerStyle((8, 2, 0))
        expected_asterisk = asterisk_style.get_path().transformed(asterisk_style.get_transform()).vertices
        assert not asterisk_style.is_filled()
        np.testing.assert_allclose(pcs[0].get_paths()[0].vertices, expected_asterisk)
        np.testing.assert_allclose(pcs[1].get_paths()[0].vertices, expected_asterisk)
    finally:
        plt.close(fig)

    # MATLAB 'p'/'pentagram' (5-point star) and 'h'/'hexagram' (6-point star): hollow in emarker, filled in emarker2.
    for sym, expected_points in (("p", 5), ("pentagram", 5), ("h", 6), ("hexagram", 6)):
        fig, ax = plt.subplots()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                topoplot(data, locs, axes=ax, emarker=[sym, "b", 10, 1.5], emarker2=[[1], sym, "r", 12, 1.5])
                fig.canvas.draw()
            assert not any("unfilled marker" in str(item.message) for item in caught)
            pcs = sorted((c for c in ax.collections if isinstance(c, PathCollection)), key=lambda c: c.get_zorder())
            assert len(pcs) == 2
            emarker_col, emarker2_col = pcs
            assert np.asarray(emarker_col.get_facecolor()).size == 0  # hollow outline in emarker
            np.testing.assert_allclose(emarker_col.get_edgecolor()[0], [0.0, 0.0, 1.0, 1.0])
            np.testing.assert_allclose(emarker2_col.get_facecolor()[0], [1.0, 0.0, 0.0, 1.0])  # filled in emarker2
            # 5-point star has 10 alternating outer/inner vertices (+ closing); 6-point star has 12 (+ closing).
            verts = emarker2_col.get_paths()[0].vertices
            assert len(verts) in (2 * expected_points, 2 * expected_points + 1)
        finally:
            plt.close(fig)


def test_topoplot_emarker2_indexing_plotchans_plotrad_and_electrodes_modes():
    locs = [
        {"labels": "Fz", "theta": 0.0, "radius": 0.3},
        {"labels": "Cz", "theta": 0.0, "radius": 0.0},
        {"labels": "Pz", "theta": 180.0, "radius": 0.3},
        {"labels": "C3", "theta": 270.0, "radius": 0.4},
        {"labels": "C4", "theta": 90.0, "radius": 0.4},
    ]
    data = np.array([1.0, 2.0, -1.0, 0.5, -0.5])

    # Matches MATLAB topoplot.m:913-914: emarker2 indexes into post-plotchans pltchans.
    fig, ax = plt.subplots()
    try:
        topoplot(
            data,
            locs,
            axes=ax,
            plotchans=[2, 4, 5],
            emarker=["s", "b", 8, 2],
            emarker2=[[2], "o", [0.0, 1.0, 0.0], 14, 3],
        )
        pcs = sorted((c for c in ax.collections if isinstance(c, PathCollection)), key=lambda c: c.get_zorder())
        assert len(pcs) == 2
        normal_col, mark2_col = pcs
        assert normal_col.get_zorder() < mark2_col.get_zorder()
        assert len(normal_col.get_offsets()) == 2  # Cz and C4
        assert len(mark2_col.get_offsets()) == 1  # C3 (position 2 of plotchans)
        assert mark2_col.get_offsets()[0, 0] < 0  # C3 is on the left
        np.testing.assert_allclose(mark2_col.get_facecolor()[0], [0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(mark2_col.get_sizes(), [196.0])
        np.testing.assert_allclose(mark2_col.get_linewidth(), [3.0])
        assert np.asarray(normal_col.get_facecolor()).size == 0  # hollow square
    finally:
        plt.close(fig)

    # Channel excluded by plotrad is not drawn by emarker2; later channels keep their 1-based position.
    wide_locs = [dict(loc) for loc in locs]
    wide_locs[2]["radius"] = 0.8  # Pz outside plotrad=0.5
    fig, ax = plt.subplots()
    try:
        topoplot(data, wide_locs, axes=ax, plotrad=0.5, electrodes="off", emarker2=[[3, 4], "d", "r", 11, 2])
        pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(pcs) == 1
        assert len(pcs[0].get_offsets()) == 1  # only C3 (4); Pz (3) excluded by plotrad
        assert pcs[0].get_offsets()[0, 0] < 0
    finally:
        plt.close(fig)

    # electrodes='off' suppresses emarker but STILL draws emarker2 (topoplot.m:1614-1616).
    fig, ax = plt.subplots()
    try:
        topoplot(data, locs, axes=ax, electrodes="off", emarker2=[[1, 3], "d", "m", 12, 2])
        pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(pcs) == 1
        assert len(pcs[0].get_offsets()) == 2
        np.testing.assert_allclose(pcs[0].get_sizes(), [144.0])
    finally:
        plt.close(fig)

    # electrodes='labels' / 'numbers' draw text only (neither emarker nor emarker2).
    for mode, expected_texts in (("labels", ["Fz", "Cz", "Pz", "C3", "C4"]), ("nums", ["1", "2", "3", "4", "5"])):
        fig, ax = plt.subplots()
        try:
            topoplot(data, locs, axes=ax, electrodes=mode, emarker2=[[1], "o", "r", 12, 1])
            assert not any(isinstance(c, PathCollection) for c in ax.collections)
            assert [t.get_text() for t in ax.texts] == expected_texts
        finally:
            plt.close(fig)

    # electrodes='ptsnumbers' on interpolated map draws both markers and original 1-based channel numbers.
    fig, ax = plt.subplots()
    try:
        topoplot(data, locs, axes=ax, plotchans=[2, 4, 5], electrodes="ptsnumbers")
        pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(pcs) == 1
        assert len(pcs[0].get_offsets()) == 3
        assert [t.get_text() for t in ax.texts] == ["2", "4", "5"]
    finally:
        plt.close(fig)

    # Default electrodes disables markers for >64 channels only when electrodes is omitted.
    many_locs = [{"labels": f"E{i + 1}", "theta": i * (360.0 / 68), "radius": 0.4 if i else 0.0} for i in range(68)]
    many_data = np.linspace(-1.0, 1.0, 68)
    fig, ax = plt.subplots()
    try:
        topoplot(many_data, many_locs, axes=ax)
        assert not any(isinstance(c, PathCollection) for c in ax.collections)
    finally:
        plt.close(fig)

    # Unlocated channels (empty theta/radius, e.g. EOG1/EOG2) are dropped before emarker2 indexing (topoplot.m:694-915).
    locs_with_eog = [
        {"labels": "Fz", "theta": 0.0, "radius": 0.3},
        {"labels": "EOG1", "theta": np.empty((0, 0)), "radius": np.empty((0, 0))},
        {"labels": "C3", "theta": -90.0, "radius": 0.4},
        {"labels": "C4", "theta": 90.0, "radius": 0.4},
    ]
    data_with_eog = np.array([1.0, 99.0, -1.0, 0.5])
    fig, ax = plt.subplots()
    try:
        topoplot(data_with_eog, locs_with_eog, axes=ax, electrodes="off", emarker2=[[2], "o", "r", 12, 2])
        pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(pcs) == 1
        assert len(pcs[0].get_offsets()) == 1
        # Located channels are [Fz, C3, C4], so emarker2 index 2 marks C3 on the left (screen_x < 0).
        assert pcs[0].get_offsets()[0, 0] < 0
    finally:
        plt.close(fig)

    # emarker2 on style='blank' raises ValueError matching EEGLAB even with noplot='on'.
    with pytest.raises(ValueError, match="emarker2 not defined for style 'blank'"):
        topoplot([], locs, style="blank", noplot="on", emarker2=[[1]])


def test_topoplot_blank_marks_selected_channels_with_emarkersize1chan():
    locs = [
        {"labels": "Fz", "theta": 0.0, "radius": 0.3},
        {"labels": "Cz", "theta": 0.0, "radius": 0.0},
        {"labels": "Pz", "theta": 180.0, "radius": 0.3},
        {"labels": "C3", "theta": 270.0, "radius": 0.4},
        {"labels": "C4", "theta": 90.0, "radius": 0.4},
    ]

    # Single channel index with emarkersize1chan draws one red '.' marker sized by (2 * pt / 3) ** 2.
    fig, ax = plt.subplots()
    try:
        topoplot(4, locs, style="blank", electrodes="off", emarkersize1chan=12, axes=ax)
        pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(pcs) == 1
        assert len(pcs[0].get_offsets()) == 1
        assert pcs[0].get_offsets()[0, 0] < 0  # C3 on left
        np.testing.assert_allclose(pcs[0].get_sizes(), [(2.0 * 12.0 / 3.0) ** 2])
        np.testing.assert_allclose(pcs[0].get_linewidth(), [0.0])
        np.testing.assert_allclose(pcs[0].get_facecolor()[0], [1.0, 0.0, 0.0, 1.0])
    finally:
        plt.close(fig)

    # Alias emarkersizemark and default size 20 pt work, and electrodes='on' keeps normal dots below.
    fig, ax = plt.subplots()
    try:
        topoplot([1, 5], locs, style="blank", electrodes="on", emarkersizemark=15, axes=ax)
        pcs = sorted((c for c in ax.collections if isinstance(c, PathCollection)), key=lambda c: c.get_zorder())
        assert len(pcs) == 2
        assert len(pcs[0].get_offsets()) == 5
        assert len(pcs[1].get_offsets()) == 2
        np.testing.assert_allclose(pcs[1].get_sizes(), [(2.0 * 15.0 / 3.0) ** 2])
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        topoplot(5, locs, style="blank", electrodes="off", axes=ax)
        pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(pcs) == 1
        np.testing.assert_allclose(pcs[0].get_sizes(), [(2.0 * 20.0 / 3.0) ** 2])  # default EMARKERSIZE1CHAN=20
    finally:
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
