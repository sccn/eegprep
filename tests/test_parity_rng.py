"""
MATLAB-compatible rounding used by the rand()-based parity RNG mechanism.

Parity code uses seed 5489 with rand() (which matches MATLAB's MT19937 stream)
plus round_mat() and custom sampling (see ransac.py:rand_sample).
"""

import unittest
from eegprep.functions.miscfunc.misc import round_mat


class TestRNGIsolation(unittest.TestCase):
    """Demonstrate RNG mechanism without MATLAB dependency."""

    def test_round_mat_tie_breaking(self):
        """Test round_mat's tie-breaking behavior (rounds away from zero)."""
        # MATLAB rounds ties (.5) away from zero
        # Python's round() rounds ties to even (banker's rounding)
        # round_mat should match MATLAB

        self.assertEqual(round_mat(0.5), 1.0)  # Round up
        self.assertEqual(round_mat(-0.5), -1.0)  # Round down (away from zero)
        self.assertEqual(round_mat(1.5), 2.0)  # Round up
        self.assertEqual(round_mat(-1.5), -2.0)  # Round down (away from zero)
        self.assertEqual(round_mat(2.5), 3.0)  # Round up


if __name__ == '__main__':
    unittest.main()
