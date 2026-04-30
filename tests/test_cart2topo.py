import os
import tempfile
import unittest

import numpy as np
import scipy.io

from eegprep.cart2topo import cart2topo


class TestCart2Topo(unittest.TestCase):
    def test_cardinal_points(self):
        xyz = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        theta, radius = cart2topo(xyz)

        np.testing.assert_allclose(theta, [0.0, -180.0, -90.0, 90.0, -0.0, -0.0], atol=1e-12)
        np.testing.assert_allclose(radius, [0.5, 0.5, 0.5, 0.5, 0.0, 1.0], atol=1e-12)

    def test_single_xyz_vector(self):
        theta, radius = cart2topo([0.0, 1.0, 0.0])

        self.assertEqual(theta.shape, (1,))
        np.testing.assert_allclose(theta, [-90.0], atol=1e-12)
        np.testing.assert_allclose(radius, [0.5], atol=1e-12)

    def test_matrix_input(self):
        theta, radius = cart2topo(np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]]))

        np.testing.assert_allclose(theta, [-45.0, 45.0], atol=1e-12)
        np.testing.assert_allclose(radius, [0.3040867239846964, 0.5], atol=1e-12)

    def test_separate_coordinate_vectors(self):
        theta, radius = cart2topo([1.0, 0.0], [0.0, 1.0], [0.0, 0.0])

        np.testing.assert_allclose(theta, [0.0, -90.0], atol=1e-12)
        np.testing.assert_allclose(radius, [0.5, 0.5], atol=1e-12)

    def test_empty_input_returns_empty_arrays(self):
        theta, radius = cart2topo([])

        self.assertEqual(theta.shape, (0,))
        self.assertEqual(radius.shape, (0,))
        self.assertEqual(theta.dtype, np.float64)
        self.assertEqual(radius.dtype, np.float64)

    def test_invalid_shapes_raise(self):
        with self.assertRaisesRegex(ValueError, "N x 3"):
            cart2topo(np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "all be provided"):
            cart2topo([1.0], [2.0])
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            cart2topo([1.0, 2.0], [1.0], [0.0, 0.0])

    def test_nan_input_propagates(self):
        theta, radius = cart2topo([[np.nan, 0.0, 1.0]])

        self.assertTrue(np.isnan(theta[0]))
        self.assertTrue(np.isnan(radius[0]))

    def test_zero_vector_matches_eeglab_convention(self):
        theta, radius = cart2topo([0.0, 0.0, 0.0])

        np.testing.assert_allclose(theta, [0.0], atol=1e-12)
        np.testing.assert_allclose(radius, [0.5], atol=1e-12)

    def test_top_level_export(self):
        from eegprep import cart2topo as exported_cart2topo

        self.assertIs(exported_cart2topo, cart2topo)


@unittest.skipIf(os.getenv("EEGPREP_SKIP_MATLAB") == "1", "MATLAB not available")
class TestCart2TopoParity(unittest.TestCase):
    def setUp(self):
        try:
            from eegprep.eeglabcompat import get_eeglab

            self.eeglab = get_eeglab("MAT")
        except Exception as exc:
            self.skipTest(f"MATLAB not available: {exc}")

    def _matlab_cart2topo(self, xyz):
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as input_file:
            input_path = input_file.name
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as output_file:
            output_path = output_file.name
        try:
            scipy.io.savemat(input_path, {"xyz": xyz})
            self.eeglab.engine.eval(
                (
                    f"args = load('{input_path}'); "
                    "[theta, radius] = cart2topo(args.xyz); "
                    f"save('-mat', '{output_path}', 'theta', 'radius');"
                ),
                nargout=0,
            )
            out = scipy.io.loadmat(output_path)
            theta = np.asarray(out["theta"], dtype=float).reshape(-1)
            radius = np.asarray(out["radius"], dtype=float).reshape(-1)
            return theta, radius
        finally:
            for filename in (input_path, output_path):
                try:
                    os.remove(filename)
                except OSError:
                    pass

    def test_parity_with_eeglab_matrix_input(self):
        xyz = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.3, -0.4, 0.5],
                [0.3, -0.4, -0.5],
            ],
            dtype=float,
        )

        theta_py, radius_py = cart2topo(xyz)
        theta_ml, radius_ml = self._matlab_cart2topo(xyz)
        np.testing.assert_allclose(theta_py, theta_ml, atol=1e-12, rtol=0)
        np.testing.assert_allclose(radius_py, radius_ml, atol=1e-12, rtol=0)

    def test_parity_with_eeglab_separate_vector_input(self):
        x = np.array([1.0, 0.0, 0.3], dtype=float)
        y = np.array([0.0, 1.0, -0.4], dtype=float)
        z = np.array([0.0, 0.0, -0.5], dtype=float)
        xyz = np.column_stack([x, y, z])

        theta_py, radius_py = cart2topo(x, y, z)
        theta_ml, radius_ml = self._matlab_cart2topo(xyz)
        np.testing.assert_allclose(theta_py, theta_ml, atol=1e-12, rtol=0)
        np.testing.assert_allclose(radius_py, radius_ml, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
