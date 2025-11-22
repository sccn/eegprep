"""Test to debug MATLAB path issues in CI."""
import unittest
import os
import sys
import subprocess


class TestMatlabPath(unittest.TestCase):
    """Test MATLAB availability and path configuration."""
    
    def test_matlab_in_path(self):
        """Check if matlab executable is in PATH."""
        result = subprocess.run(['which', 'matlab'], capture_output=True, text=True)
        print(f"which matlab: {result.stdout}")
        print(f"stderr: {result.stderr}")
        print(f"returncode: {result.returncode}")
        self.assertEqual(result.returncode, 0, "MATLAB not found in PATH")
    
    def test_matlab_version(self):
        """Try to run matlab -batch version."""
        try:
            result = subprocess.run(
                ['matlab', '-batch', 'version'], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            print(f"matlab version output: {result.stdout}")
            print(f"stderr: {result.stderr}")
            print(f"returncode: {result.returncode}")
        except FileNotFoundError as e:
            self.fail(f"MATLAB executable not found: {e}")
        except subprocess.TimeoutExpired:
            self.fail("MATLAB command timed out")
    
    def test_python_matlab_engine(self):
        """Test if Python MATLAB engine can be imported."""
        try:
            import matlab.engine
            print(f"MATLAB engine module found at: {matlab.engine.__file__}")
        except ImportError as e:
            self.fail(f"Cannot import matlab.engine: {e}")
    
    def test_start_matlab_engine(self):
        """Test if MATLAB engine can be started."""
        try:
            import matlab.engine
            print("Attempting to start MATLAB engine...")
            eng = matlab.engine.start_matlab()
            print("MATLAB engine started successfully!")
            result = eng.eval('version', nargout=1)
            print(f"MATLAB version: {result}")
            eng.quit()
        except ImportError as e:
            self.skipTest(f"matlab.engine not available: {e}")
        except Exception as e:
            self.fail(f"Failed to start MATLAB engine: {e}")
    
    def test_get_eeglab_oct(self):
        """Test get_eeglab with Octave runtime."""
        try:
            from eegprep.eeglabcompat import get_eeglab
            print("Attempting to get EEGLAB with Octave runtime...")
            eeglab = get_eeglab('OCT')
            print(f"EEGLAB (Octave) loaded successfully: {type(eeglab)}")
        except Exception as e:
            print(f"Failed to load EEGLAB with Octave: {e}")
            # Don't fail the test, just report
    
    def test_get_eeglab_mat(self):
        """Test get_eeglab with MATLAB runtime."""
        try:
            from eegprep.eeglabcompat import get_eeglab
            print("Attempting to get EEGLAB with MATLAB runtime...")
            eeglab = get_eeglab('MAT')
            print(f"EEGLAB (MATLAB) loaded successfully: {type(eeglab)}")
        except ImportError as e:
            self.skipTest(f"MATLAB engine not available: {e}")
        except Exception as e:
            self.fail(f"Failed to load EEGLAB with MATLAB: {e}")
    
    def test_environment_vars(self):
        """Print relevant environment variables."""
        print("\n=== Environment Variables ===")
        for key in ['PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'MATLABROOT']:
            value = os.environ.get(key, 'NOT SET')
            print(f"{key}: {value}")
        print("=" * 40)


if __name__ == '__main__':
    unittest.main(verbosity=2)
