"""Testing utilities."""

import sys
import unittest

import numpy as np

__all__ = ['compare_eeg', 'DebuggableTestCase', 'is_debug']


# default to True since the round-tripping through file can force data to
# 32-bit precision depending on user settings etc
default_32_bit = True

# works around an issue where pop_loadset can at times read back a 2d array@
# as 3d
flatten_to_2d = True


def compare_eeg(a, b, rtol=0, atol=1e-7, use_32_bit=default_32_bit, err_msg=''):
    """Compare EEG time series data, with optional 32-bit precision."""
    if use_32_bit:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
    if flatten_to_2d:
        if a.ndim >= 3:
            a = a.reshape(a.shape[:2])
        if b.ndim >= 3:
            b = b.reshape(b.shape[:2])

    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=err_msg)


class DebuggableTestCase(unittest.TestCase):
    """Base class for test cases where exceptions can be caught in the debugger.
    This is used as follows: add a if __name__ == '__main__' block to your test
    module, and for each test case, add a line like MyTestCase.debugTestCase()
    there. Then run the module not as a unit test (Python tests in ...) but
    instead create a launch configuration that runs the module as a regular Python
    module, and run that in the debugger.

    """
    @classmethod
    def debugTestCase(cls):
        loader = unittest.defaultTestLoader
        testSuite = loader.loadTestsFromTestCase(cls)
        testSuite.debug()

def is_debug():
    """Determine whether Python is running in debug mode."""
    return getattr(sys, 'gettrace', None)() is not None
