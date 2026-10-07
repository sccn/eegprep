import unittest
import numpy as np
from unittest.mock import patch
from io import StringIO

from eegprep.utils.testing import compare_eeg


class TestCompareEeg(unittest.TestCase):
    def test_compare_eeg_different_shapes_error(self):
        """Test error when arrays have different shapes."""
        a = np.array([[1.0, 2.0]])
        b = np.array([[1.0], [2.0]])

        with self.assertRaises(ValueError) as cm:
            compare_eeg(a, b)
        self.assertIn("different shapes", str(cm.exception))

    def test_compare_eeg_tolerance_parameters(self):
        """Test rtol and atol tolerance parameters."""
        a = np.array([[1.0, 2.0]])
        b = np.array([[1.001, 2.001]])  # Small differences

        with patch('sys.stdout', StringIO()):
            # Should pass with loose tolerance
            compare_eeg(a, b, rtol=1e-2, atol=1e-2)

        # Should fail with tight tolerance
        with self.assertRaises(AssertionError):
            with patch('sys.stdout', StringIO()):
                compare_eeg(a, b, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
