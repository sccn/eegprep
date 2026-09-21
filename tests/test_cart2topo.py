import unittest

import numpy as np

from eegprep import cart2topo
from tests.eeglab_tests import eeglab_test


class TestCart2Topo(unittest.TestCase):
    @eeglab_test(
        "unittesting_sigprocfunc/cart2topo/sigprocfunc_cart2topo_wrapperTest.m",
        "test_pass_xy",
    )
    def test_diagonal_cartesian_points_match_upstream_orientation(self):
        xyz = np.asarray([[0, 1, 0], [0, -1, 0], [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]])

        theta, radius, *_ = cart2topo(xyz)

        np.testing.assert_allclose(theta, [-90, 90, -45, -135, 45, 135])
        np.testing.assert_allclose(radius, np.full(6, 0.5))

    @eeglab_test(
        "unittesting_sigprocfunc/cart2topo/sigprocfunc_cart2topo_wrapperTest.m",
        "test_pass_zero_y",
    )
    def test_near_zero_y_preserves_upstream_signed_angles(self):
        xyz = np.asarray([[1, 0.000001000001, 0], [-1, 0.000001000001, 0]])

        theta, radius, *_ = cart2topo(xyz)

        np.testing.assert_allclose(theta, [0, -180], atol=1e-4)
        np.testing.assert_allclose(radius, [0.5, 0.5])

    def test_cardinal_cartesian_points_match_eeglab_orientation(self):
        xyz = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )

        theta, radius, x, y, z = cart2topo(xyz)

        np.testing.assert_allclose(theta, [0.0, -180.0, -90.0, 90.0])
        np.testing.assert_allclose(radius, [0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(x, xyz[:, 0])
        np.testing.assert_allclose(y, xyz[:, 1])
        np.testing.assert_allclose(z, xyz[:, 2])

    def test_vertex_and_lower_plane_radius(self):
        theta, radius, *_ = cart2topo([0.0, 0.0], [0.0, 0.0], [1.0, -1.0])

        np.testing.assert_allclose(theta, [0.0, 0.0])
        np.testing.assert_allclose(radius, [0.0, 1.0])

    def test_rejects_legacy_optional_arguments(self):
        with self.assertRaisesRegex(ValueError, "Additional cart2topo parameters"):
            cart2topo(np.ones((2, 3)), "center", [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
