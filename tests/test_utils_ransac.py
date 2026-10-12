import unittest
import numpy as np

from eegprep.plugins.clean_rawdata.private.ransac import rand_permutation, rand_sample, calc_projector
from eegprep.plugins.clean_rawdata.private.sphericalSplineInterpolate import sphericalSplineInterpolate


def _scalar_rand_sample(n, m, stream):
    pool = np.arange(n)
    for k in range(m):
        remaining = n - k
        choice = int(np.floor((remaining - 1) * stream.rand() + 0.5))
        idx = k + choice
        pool[k], pool[idx] = pool[idx], pool[k]
    return pool[:m].copy()


def _scalar_rand_permutation(n, stream):
    result = np.arange(n)
    for k in range(n - 1, 0, -1):
        j = int(np.floor(k * stream.rand() + 0.5))
        result[k], result[j] = result[j], result[k]
    return result


class TestRandSample(unittest.TestCase):
    """Test the rand_sample function for random sampling without replacement."""

    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.RandomState(42)  # Fixed seed for reproducibility

    def test_vectorized_rng_matches_scalar_reference(self):
        """Test that batched random draws preserve the scalar RNG sequence."""
        for seed, n, m in ((42, 10, 5), (5489, 20, 20), (123, 8, 0)):
            with self.subTest(seed=seed, n=n, m=m):
                vectorized_rng = np.random.RandomState(seed)
                scalar_rng = np.random.RandomState(seed)

                result = rand_sample(n, m, vectorized_rng)
                expected = _scalar_rand_sample(n, m, scalar_rng)

                np.testing.assert_array_equal(result, expected)
                np.testing.assert_allclose(vectorized_rng.rand(3), scalar_rng.rand(3))


class TestRandPermutation(unittest.TestCase):
    """Test the rand_permutation function for MATLAB-parity shuffling."""

    def test_vectorized_rng_matches_scalar_reference(self):
        """Test that batched random draws preserve the scalar RNG sequence."""
        for seed, n in ((42, 10), (5489, 20), (123, 1), (321, 0)):
            with self.subTest(seed=seed, n=n):
                vectorized_rng = np.random.RandomState(seed)
                scalar_rng = np.random.RandomState(seed)

                result = rand_permutation(n, vectorized_rng)
                expected = _scalar_rand_permutation(n, scalar_rng)

                np.testing.assert_array_equal(result, expected)
                np.testing.assert_allclose(vectorized_rng.rand(3), scalar_rng.rand(3))


class TestCalcProjector(unittest.TestCase):
    """Test the calc_projector function for RANSAC reconstruction matrices."""

    def setUp(self):
        """Set up test fixtures with synthetic channel locations."""
        # Create synthetic 3D channel locations (spherical coordinates)
        self.n_channels = 8
        theta = np.linspace(0, 2 * np.pi, self.n_channels, endpoint=False)
        phi = np.pi / 4  # Fixed elevation

        self.locs = np.column_stack(
            [np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi) * np.ones(self.n_channels)]
        )

        # Test parameters
        self.num_samples = 5
        self.subset_size = 4
        self.rng = np.random.RandomState(12345)

    def test_real_interpolation_channel_mapping_and_assembly(self):
        """Validate calc_projector against the real spherical-spline kernel.

        The mock-based tests above only check output shape and call counts, so a
        bug that permutes channel indices or mishandles the per-subset transpose
        would pass. This test runs the real interpolation on a small montage and
        independently reproduces each subset's reconstruction matrix, catching
        such assembly bugs.
        """
        num_samples = 4
        subset_size = self.n_channels - 2

        # Reproduce the exact subsets calc_projector samples (k from num_samples-1..0).
        subset_stream = np.random.RandomState(7)
        subsets = {k: rand_sample(self.n_channels, subset_size, subset_stream) for k in range(num_samples - 1, -1, -1)}

        projector = calc_projector(self.locs, num_samples, subset_size, stream=np.random.RandomState(7))

        # Output must be a finite, real-valued bag of reconstruction matrices.
        self.assertEqual(projector.shape, (self.n_channels, self.n_channels * num_samples))
        self.assertTrue(np.isrealobj(projector))
        self.assertTrue(np.all(np.isfinite(projector)))

        blocks = projector.reshape(self.n_channels, num_samples, self.n_channels)
        for k, sample in subsets.items():
            block = blocks[:, k, :]

            # Only the rows of the sampled source channels carry weight; the two
            # unsampled channels must stay all-zero. A channel-index permutation
            # would shift the zero rows away from the unsampled channels.
            nonzero_rows = np.flatnonzero(np.any(block != 0, axis=1))
            np.testing.assert_array_equal(np.sort(nonzero_rows), np.sort(sample))

            # The non-zero rows must equal the real spherical-spline weights for
            # this subset, transposed exactly as calc_projector assembles them.
            expected_w = sphericalSplineInterpolate(self.locs[sample, :].T, self.locs.T)[0]
            np.testing.assert_allclose(block[sample, :], np.real(expected_w).T, rtol=1e-10, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
