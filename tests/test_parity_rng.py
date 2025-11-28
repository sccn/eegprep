"""
Test parity of random number generation between Python and MATLAB.

IMPORTANT: The RNG mechanism works as follows:
1. Both Python and MATLAB use seed 5489 (MATLAB default)
2. Both use the Mersenne Twister algorithm (MT19937)
3. The rand() uniform distribution DOES produce the same sequence
4. However, randn() normal distribution does NOT match between implementations
5. For parity, use rand() + round_mat() + custom sampling (see ransac.py:rand_sample)

This mechanism is used throughout the codebase (e.g., clean_channels.py:111,
eeg_picard.py:47, ransac.py:9-31) to ensure reproducible results.
"""

import os
import unittest
import numpy as np
import tempfile
import scipy.io
from eegprep.eeglabcompat import get_eeglab
from eegprep.utils.ransac import rand_sample
from eegprep.utils.misc import round_mat


class TestRNGParity(unittest.TestCase):
    """Test that Python and MATLAB produce identical random sequences using rand()."""

    def setUp(self):
        """Set up test fixtures."""
        # Try to get MATLAB engine
        try:
            self.eeglab = get_eeglab('MAT', auto_file_roundtrip=False)
            self.matlab_available = True
        except Exception as e:
            self.matlab_available = False
            self.skipTest(f"MATLAB not available: {e}")

    def test_rng_uniform_parity(self):
        """Test that rand() (uniform) produces the SAME ORDERED sequence (1D) in Python and MATLAB."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        seed = 5489  # MATLAB default

        # Python random sequence (uniform) - 1D to avoid column/row major issues
        rng_py = np.random.RandomState(seed)
        py_rand = rng_py.rand(50)  # 1D array

        # MATLAB random sequence (uniform) - force 1D
        temp_file = tempfile.mktemp(suffix='.mat')
        matlab_code = f"""
        rng({seed}, 'twister');
        ml_rand = rand(1, 50);  % Row vector to ensure 1D
        save('{temp_file}', 'ml_rand');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB results
        mat_data = scipy.io.loadmat(temp_file)
        ml_rand = mat_data['ml_rand'].flatten()

        # Clean up
        os.remove(temp_file)

        # Compare rand (uniform distribution) - THIS SHOULD MATCH for 1D
        print(f"\nrand() 1D uniform comparison:")
        print(f"  Python first 5 values: {py_rand[:5]}")
        print(f"  MATLAB first 5 values: {ml_rand[:5]}")
        print(f"  Max absolute diff: {np.max(np.abs(py_rand - ml_rand)):.2e}")

        np.testing.assert_allclose(py_rand, ml_rand, rtol=1e-15, atol=1e-15,
                                   err_msg="rand() 1D uniform should produce identical sequences")

    def test_rng_normal_incompatibility(self):
        """Test that randn() (normal) produces DIFFERENT sequences (known incompatibility)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        seed = 5489

        # Python random sequence (normal)
        rng_py = np.random.RandomState(seed)
        py_randn = rng_py.randn(10, 5)

        # MATLAB random sequence (normal)
        temp_file = tempfile.mktemp(suffix='.mat')
        matlab_code = f"""
        rng({seed}, 'twister');
        ml_randn = randn(10, 5);
        save('{temp_file}', 'ml_randn');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB results
        mat_data = scipy.io.loadmat(temp_file)
        ml_randn = mat_data['ml_randn']

        # Clean up
        os.remove(temp_file)

        # Compare randn (normal distribution) - THIS SHOULD DIFFER
        print(f"\nrandn() normal comparison (EXPECTED TO DIFFER):")
        print(f"  Python first 3 values: {py_randn.flatten()[:3]}")
        print(f"  MATLAB first 3 values: {ml_randn.flatten()[:3]}")
        print(f"  Max absolute diff: {np.max(np.abs(py_randn - ml_randn)):.2e}")

        are_different = not np.allclose(py_randn, ml_randn, rtol=1e-10, atol=1e-10)
        self.assertTrue(are_different,
                       "randn() normal distribution differs between Python and MATLAB "
                       "(this is expected - use rand() for parity)")

    def test_rand_sample_mechanism(self):
        """Test the rand_sample mechanism that provides MATLAB parity."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        seed = 5489
        n = 20  # sample from 20 items
        m = 5   # select 5 items

        # Python rand_sample (from ransac.py)
        rng_py = np.random.RandomState(seed)
        py_sample = rand_sample(n, m, rng_py)

        # MATLAB equivalent (manual implementation of rand_sample logic)
        temp_file = tempfile.mktemp(suffix='.mat')
        matlab_code = f"""
        rng({seed}, 'twister');
        n = {n};
        m = {m};
        pool = 0:(n-1);  % MATLAB 0-indexed to match Python
        result = zeros(1, m);

        for k = 1:m
            choice = round((length(pool) - 1) * rand()) + 1;  % MATLAB 1-indexed
            result(k) = pool(choice);
            pool(choice) = [];
        end

        ml_sample = result;
        save('{temp_file}', 'ml_sample');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB results
        mat_data = scipy.io.loadmat(temp_file)
        ml_sample = mat_data['ml_sample'].flatten().astype(int)

        # Clean up
        os.remove(temp_file)

        # Compare
        print(f"\nrand_sample comparison:")
        print(f"  Python sample: {py_sample}")
        print(f"  MATLAB sample: {ml_sample}")

        np.testing.assert_array_equal(py_sample, ml_sample,
                                     err_msg="rand_sample should produce identical results")

    def test_round_mat_parity(self):
        """Test that round_mat matches MATLAB's round() behavior."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Test values including tie-breaking cases
        test_values = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49, 0.51])

        # Python round_mat
        py_rounded = np.array([round_mat(x) for x in test_values])

        # MATLAB round
        temp_file = tempfile.mktemp(suffix='.mat')
        matlab_code = f"""
        test_values = {list(test_values)};
        ml_rounded = round(test_values);
        save('{temp_file}', 'ml_rounded');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB results
        mat_data = scipy.io.loadmat(temp_file)
        ml_rounded = mat_data['ml_rounded'].flatten()

        # Clean up
        os.remove(temp_file)

        # Compare
        print(f"\nround_mat comparison:")
        print(f"  Test values:  {test_values}")
        print(f"  Python round: {py_rounded}")
        print(f"  MATLAB round: {ml_rounded}")

        np.testing.assert_array_equal(py_rounded, ml_rounded,
                                     err_msg="round_mat should match MATLAB round()")

    def test_rng_permutation_compatibility(self):
        """Test that permutation differs (known incompatibility)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        seed = 5489
        n = 20

        # Python permutation
        rng_py = np.random.RandomState(seed)
        py_perm = rng_py.permutation(n)

        # MATLAB permutation
        temp_file = tempfile.mktemp(suffix='.mat')
        matlab_code = f"""
        rng({seed}, 'twister');
        ml_perm = randperm({n});
        save('{temp_file}', 'ml_perm');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB results
        mat_data = scipy.io.loadmat(temp_file)
        ml_perm = mat_data['ml_perm'].flatten()

        # Clean up
        os.remove(temp_file)

        # Check if they differ (they should - different algorithms)
        are_different = not np.array_equal(py_perm, ml_perm)

        print(f"\nPermutation comparison (EXPECTED TO DIFFER):")
        print(f"  Python permutation: {py_perm[:5]}...")
        print(f"  MATLAB permutation: {ml_perm[:5]}...")
        print(f"  Are different: {are_different}")

        # Document this known incompatibility
        self.assertTrue(are_different,
                       "Permutation algorithms differ between Python and MATLAB "
                       "(this is expected - randperm uses different algorithm than permutation)")


class TestRNGIsolation(unittest.TestCase):
    """Demonstrate RNG mechanism without MATLAB dependency."""

    def test_python_rng_deterministic(self):
        """Test that Python RNG is deterministic with same seed."""
        seed = 5489

        # First run
        rng1 = np.random.RandomState(seed)
        seq1 = rng1.rand(100)  # Use rand() for consistency

        # Second run with same seed
        rng2 = np.random.RandomState(seed)
        seq2 = rng2.rand(100)

        # Should be identical
        np.testing.assert_array_equal(seq1, seq2,
                                     err_msg="Same seed should produce identical sequence")

    def test_python_rng_different_seeds(self):
        """Test that different seeds produce different sequences."""
        rng1 = np.random.RandomState(5489)
        seq1 = rng1.rand(100)

        rng2 = np.random.RandomState(12345)
        seq2 = rng2.rand(100)

        # Should be different
        self.assertFalse(np.array_equal(seq1, seq2),
                        "Different seeds should produce different sequences")

    def test_matlab_default_seed_value(self):
        """Document that 5489 is MATLAB's default seed."""
        # This is the seed value used throughout the codebase:
        # - clean_channels.py:111
        # - eeg_picard.py:47
        # - Corresponds to MATLAB's rng('default')

        matlab_default_seed = 5489

        # Create RNG with this seed
        rng = np.random.RandomState(matlab_default_seed)
        first_uniform_value = rng.rand()

        # Expected first value when using seed 5489
        # (Verified against MATLAB: rng(5489,'twister'); rand)
        expected_first_uniform_value = 0.8147236863931789  # Matches MATLAB output

        # Reset and check
        rng = np.random.RandomState(matlab_default_seed)
        actual_first_uniform_value = rng.rand()

        self.assertAlmostEqual(actual_first_uniform_value, expected_first_uniform_value, places=15,
                              msg="MATLAB default seed (5489) should produce expected first uniform value")

    def test_rand_sample_deterministic(self):
        """Test that rand_sample is deterministic."""
        seed = 5489
        n = 20
        m = 5

        # First run
        rng1 = np.random.RandomState(seed)
        sample1 = rand_sample(n, m, rng1)

        # Second run
        rng2 = np.random.RandomState(seed)
        sample2 = rand_sample(n, m, rng2)

        # Should be identical
        np.testing.assert_array_equal(sample1, sample2,
                                     err_msg="rand_sample with same seed should be deterministic")

    def test_round_mat_tie_breaking(self):
        """Test round_mat's tie-breaking behavior (rounds away from zero)."""
        # MATLAB rounds ties (.5) away from zero
        # Python's round() rounds ties to even (banker's rounding)
        # round_mat should match MATLAB

        self.assertEqual(round_mat(0.5), 1.0)   # Round up
        self.assertEqual(round_mat(-0.5), -1.0)  # Round down (away from zero)
        self.assertEqual(round_mat(1.5), 2.0)   # Round up
        self.assertEqual(round_mat(-1.5), -2.0)  # Round down (away from zero)
        self.assertEqual(round_mat(2.5), 3.0)   # Round up


if __name__ == '__main__':
    unittest.main()
