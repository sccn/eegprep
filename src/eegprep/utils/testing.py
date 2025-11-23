"""Testing utilities."""

import os
import shutil
import sys
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

__all__ = ['compare_eeg', 'DebuggableTestCase', 'is_debug', 'use_64bit_eeg_options']


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

    # check if a and b are the same shape
    if a.shape != b.shape:
        raise ValueError(f"a and b have different shapes: {a.shape} != {b.shape}")
    
    # compute and show actual differences even for 2D arrays
    if a.ndim == 2:
        a = a.flatten()
        b = b.flatten()
    actual_rtol = np.max(np.abs(a - b) / (np.abs(a) + np.abs(b)))
    actual_atol = np.max(np.abs(a - b))
    print(f"Actual differences: rtol: {actual_rtol}, atol: {actual_atol}")
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
        """Debug the test case."""
        loader = unittest.defaultTestLoader
        testSuite = loader.loadTestsFromTestCase(cls)
        testSuite.debug()

def is_debug():
    """Determine whether Python is running in debug mode."""
    return getattr(sys, 'gettrace', None)() is not None


@contextmanager
def use_64bit_eeg_options():
    """Context manager to temporarily use EEG options that preserve 64-bit precision floating-point data.

    This can be used in unit tests that compare vs. MATLAB outputs and ensure that these tests do not spuriously
    fail due to regression to single-precision floats on the MATLAB side.

    This context manager:
    - Backs up the user's ~/eeg_options.m file if it exists
    - Replaces it with the 64-bit version from resources/eeg_options_64bit.m
    - Restores the original file on cleanup (or removes it if it didn't exist)

    Usage:
        with use_64bit_eeg_options():
            # Your code that needs 64-bit EEG options
            pass
    """
    # Get paths
    homedir = Path.home()
    eeg_options_path = homedir / 'eeg_options.m'
    backup_path = homedir / 'eeg_options.m.backup'
    
    # Find the source file in the package resources (works for both installed and dev)
    package_root = Path(__file__).resolve().parent.parent
    source_path = package_root / 'resources' / 'eeg_options_64bit.m'
    
    # Check if source file exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Track whether the file existed before
    file_existed_before = eeg_options_path.exists()
    
    # Backup existing file if present
    if file_existed_before:
        shutil.copy2(eeg_options_path, backup_path)
    
    # Copy the 64-bit options file
    shutil.copy2(source_path, eeg_options_path)
    
    try:
        yield
    finally:
        # Cleanup: restore or delete as appropriate
        if file_existed_before:
            # Restore from backup
            if backup_path.exists():
                shutil.copy2(backup_path, eeg_options_path)
                backup_path.unlink()  # Remove the backup file
            else:
                warnings.warn(
                    f"Backup file {backup_path} went missing, could not restore original eeg_options.m",
                    UserWarning
                )
        else:
            # Delete the file since it didn't exist before
            if eeg_options_path.exists():
                eeg_options_path.unlink()
