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

    def test_coincident_points_average_like_matlab_griddata(self):
        theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        x, y = np.cos(theta), np.sin(theta)
        values = np.linspace(-1, 1, 8)
        query = np.linspace(-1, 1, 11)
        xq, yq = np.meshgrid(query, query)

        merged = values.copy()
        merged[0] = 0.5 * (values[0] + 3.0)
        expected = griddata_v4(x, y, merged, xq, yq)
        actual = griddata_v4(np.append(x, x[0]), np.append(y, y[0]), np.append(values, 3.0), xq, yq)

        np.testing.assert_allclose(actual, expected)

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
                self.assertIn("topoplot griddata_v4 merged 2", str(caught.warning))
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
                self.assertIn("topoplot griddata_v4 merged 2", str(caught.warning))
                expected = griddata_v4(merged[:, 0], merged[:, 1], merged[:, 2], xq, yq)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


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

    def test_plotchans_keeps_the_requested_channels(self):
        keep = [0, 5, 8]
        _, actual, _, _, _ = topoplot(self.datavector, self.chan_locs, noplot="on", plotchans=keep)
        _, expected, _, _, _ = topoplot(
            self.datavector[keep],
            [self.chan_locs[index] for index in keep],
            noplot="on",
        )
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_plotchans_outside_the_channel_list_names_the_fix(self):
        message = (
            r"topoplot: plotchans must be 0-based indices from 0 through 8\. "
            r"Got \[0, 99\]\. Pass the channel indices to include, for example \[0, 2\]\."
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


if __name__ == '__main__':
    unittest.main()
