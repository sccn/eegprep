"""Test to debug MATLAB path issues in CI.

Note: This project uses matlab.engine Python API, not the matlab CLI.
The matlab.engine finds MATLAB through its own mechanism and doesn't
require 'matlab' to be in PATH.
"""
import os
import subprocess
import unittest


class TestMatlabPath(unittest.TestCase):
    """Test MATLAB availability and path configuration."""

    def test_python_matlab_engine(self):
        """Test if Python MATLAB engine can be imported."""
        if os.getenv('EEGPREP_SKIP_MATLAB') == '1':
            self.skipTest("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
        try:
            import matlab.engine
            print(f"MATLAB engine module found at: {matlab.engine.__file__}")
        except ImportError as e:
            self.skipTest(f"matlab.engine not installed: {e}")

    def test_start_matlab_engine(self):
        """Test if MATLAB engine can be started."""
        if os.getenv('EEGPREP_SKIP_MATLAB') == '1':
            self.skipTest("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
        try:
            import matlab.engine
        except ImportError as e:
            self.skipTest(f"matlab.engine not available: {e}")

        print("Attempting to start MATLAB engine...")
        eng = matlab.engine.start_matlab()
        print("MATLAB engine started successfully!")
        result = eng.eval('version', nargout=1)
        print(f"MATLAB version: {result}")
        eng.quit()

    def test_get_eeglab_oct(self):
        """Test get_eeglab with Octave runtime (informational)."""
        try:
            from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
            print("Attempting to get EEGLAB with Octave runtime...")
            eeglab = get_eeglab('OCT')
            print(f"EEGLAB (Octave) loaded successfully: {type(eeglab)}")
        except Exception as e:
            self.skipTest(f"Octave not available: {e}")

    def test_get_eeglab_mat(self):
        """Test get_eeglab with MATLAB runtime."""
        if os.getenv('EEGPREP_SKIP_MATLAB') == '1':
            self.skipTest("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
        try:
            from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
            print("Attempting to get EEGLAB with MATLAB runtime...")
            eeglab = get_eeglab('MAT')
            print(f"EEGLAB (MATLAB) loaded successfully: {type(eeglab)}")
        except ImportError as e:
            self.skipTest(f"MATLAB engine not available: {e}")

    def test_environment_vars(self):
        """Print relevant environment variables (informational)."""
        print("\n=== Environment Variables ===")
        for key in ['PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'MATLABROOT']:
            value = os.environ.get(key, 'NOT SET')
            # Truncate long values
            if len(value) > 200:
                value = value[:200] + '...'
            print(f"{key}: {value}")
        print("=" * 40)


if __name__ == '__main__':
    unittest.main(verbosity=2)
