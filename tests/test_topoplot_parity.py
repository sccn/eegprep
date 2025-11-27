"""
Test parity between Python and MATLAB implementations of topoplot.

This test compares the Python implementation against the MATLAB/EEGLAB reference.
"""

# Disable multithreading for deterministic numerical results
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import unittest
import numpy as np
from eegprep import pop_loadset
from eegprep.topoplot import topoplot
from eegprep.eeglabcompat import get_eeglab
import tempfile
import scipy.io

local_url = os.path.join(os.path.dirname(__file__), '../data/')


class TestTopoplotParity(unittest.TestCase):
    """Test parity between Python and MATLAB topoplot implementations."""

    def setUp(self):
        """Set up test fixtures."""
        # Try to get MATLAB engine
        try:
            self.eeglab = get_eeglab('MAT', auto_file_roundtrip=False)
            self.matlab_available = True
        except Exception as e:
            self.matlab_available = False
            self.skipTest(f"MATLAB not available: {e}")

        # Load real EEG dataset with ICA
        test_file = os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')
        self.EEG = pop_loadset(test_file)

    def test_parity_single_component_noplot(self):
        """Test parity with MATLAB for single IC topography (noplot mode)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Get first IC weights
        icawinv = self.EEG['icawinv']
        datavector = icawinv[:, 0]  # First component
        chanlocs = self.EEG['chanlocs']

        # Python result
        _, Zi_py, _, _, _ = topoplot(datavector, chanlocs, noplot='on')

        # MATLAB result - need to call via file roundtrip for complex outputs
        from eegprep import pop_saveset
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        datavector = EEG.icawinv(:, 1);
        [~, Zi, ~, ~, ~] = topoplot(datavector, EEG.chanlocs, 'noplot', 'on');
        save('{temp_file}.mat', 'Zi');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        Zi_ml = mat_data['Zi']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare results
        # Max absolute diff: TBD, Mean absolute diff: TBD
        # Max relative diff: TBD, Mean relative diff: TBD
        np.testing.assert_allclose(Zi_py, Zi_ml, rtol=1e-5, atol=1e-8,
                                   err_msg="topoplot Zi results differ beyond tolerance",
                                   equal_nan=True)

    def test_parity_multiple_components(self):
        """Test parity with MATLAB for multiple IC topographies."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Test first 5 components
        icawinv = self.EEG['icawinv']
        chanlocs = self.EEG['chanlocs']

        from eegprep import pop_saveset
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        for ic_idx in range(min(5, icawinv.shape[1])):
            datavector = icawinv[:, ic_idx]

            # Python result
            _, Zi_py, _, _, _ = topoplot(datavector, chanlocs, noplot='on')

            # MATLAB result
            matlab_code = f"""
            EEG = pop_loadset('{temp_file}');
            datavector = EEG.icawinv(:, {ic_idx + 1});
            [~, Zi, ~, ~, ~] = topoplot(datavector, EEG.chanlocs, 'noplot', 'on');
            save('{temp_file}_ic{ic_idx}.mat', 'Zi');
            """
            self.eeglab.eval(matlab_code, nargout=0)

            # Load MATLAB result
            mat_data = scipy.io.loadmat(f'{temp_file}_ic{ic_idx}.mat')
            Zi_ml = mat_data['Zi']

            # Compare results
            # Max absolute diff: TBD, Mean absolute diff: TBD
            # Max relative diff: TBD, Mean relative diff: TBD
            np.testing.assert_allclose(Zi_py, Zi_ml, rtol=1e-5, atol=1e-8,
                                       err_msg=f"topoplot Zi results differ for IC {ic_idx}",
                                       equal_nan=True)

            # Clean up IC-specific file
            os.remove(f'{temp_file}_ic{ic_idx}.mat')

        # Clean up temp files
        os.remove(temp_file)
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))


if __name__ == '__main__':
    unittest.main()
