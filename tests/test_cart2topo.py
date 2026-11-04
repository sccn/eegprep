import unittest

import numpy as np

from eegprep import cart2topo
from tests.eeglab_tests import assert_matlab_near, eeglab_test


@eeglab_test("unittesting_sigprocfunc/cart2topo/sigprocfunc_cart2topo_wrapperTest.m", "test_pass_xy")
@eeglab_test("unittesting_sigprocfunc/cart2topo/sigprocfunc_cart2topo_wrapperTest.m", "test_pass_zero_y")
def test_reference_cart2topo_original_coordinate_matrices(eeglab_backend):
    for coordinates, expected in (
        ([[0, 1, 0], [0, -1, 0], [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]], [-90, 90, -45, -135, 45, 135]),
        ([[1, 0.000001000001, 0], [-1, 0.000001000001, 0]], [0, -180]),
    ):
        theta, radius = eeglab_backend("cart2topo", np.array(coordinates, dtype=float), nargout=2)
        assert_matlab_near(theta.T, np.array([expected], dtype=float).T)
        assert_matlab_near(radius.T, np.full((len(expected), 1), 0.5))


class TestCart2Topo(unittest.TestCase):
    def test_vertex_and_lower_plane_radius(self):
        theta, radius, *_ = cart2topo([0.0, 0.0], [0.0, 0.0], [1.0, -1.0])

        np.testing.assert_allclose(theta, [0.0, 0.0])
        np.testing.assert_allclose(radius, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
