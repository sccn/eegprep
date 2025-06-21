import os
import unittest
import numpy as np
from eegprep import pop_loadset, eeg_picard, pop_saveset
from eegprep.eeglabcompat import get_eeglab

# ASSESSMENT OF THE TEST RESULTS
# -----------------------------
# The current conclusion is that while MATLAB and Octave are not exactly the same, they are close enough.
# However, Python is quite different. The image is saved in the test folder of the difference between the 2-D arrays.
# More investigation is needed to understand why this is the case. The implementation are quite different to start with.


# where the test resources
local_url = os.path.join(os.path.dirname(__file__), '../data/')

class TestEegPicard(unittest.TestCase):

    def setUp(self):
        # Using a standard test file. 
        # Even if it has ICA data, picard should overwrite it.
        self.EEG = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set'))

    def test_picard_engines(self):
        """Test eeg_picard with Python, MATLAB, and Octave engines."""
        
        # --- Python Engine ---
        print("Running Picard with Python engine...")
        EEG_python = eeg_picard(self.EEG.copy())
        pop_saveset(EEG_python, os.path.join(local_url, 'eeglab_data_picard_python.set'))
        print("Python engine test completed.")

        # --- MATLAB Engine ---
        print("Running Picard with MATLAB engine...")
        eeglab_mat = get_eeglab('MAT')
        EEG_matlab = eeg_picard(self.EEG.copy(), engine=eeglab_mat)
        pop_saveset(EEG_matlab, os.path.join(local_url, 'eeglab_data_picard_matlab.set'))
        print("MATLAB engine test completed.")

        # --- Octave Engine ---
        print("Running Picard with Octave engine...")
        eeglab_oct = get_eeglab('OCT')
        EEG_octave = eeg_picard(self.EEG.copy(), engine=eeglab_oct)
        pop_saveset(EEG_octave, os.path.join(local_url, 'eeglab_data_picard_octave.set'))
        print("Octave engine test completed.")

        # --- Assertions ---
        
        # Check that all results have the necessary ICA fields
        for eeg_result, engine_name in [(EEG_python, 'Python'), (EEG_matlab, 'MATLAB'), (EEG_octave, 'Octave')]:
            with self.subTest(engine=engine_name):
                self.assertIn('icaweights', eeg_result)
                self.assertIn('icasphere', eeg_result)
                self.assertIn('icawinv', eeg_result)
                self.assertIn('icaact', eeg_result)
                self.assertIn('icachansind', eeg_result)

        # Check shapes consistency
        # Assuming number of components is equal to number of channels by default
        n_chans = self.EEG['nbchan']
        n_pnts = self.EEG['pnts']
        n_trials = self.EEG['trials']
        
        all_results = {
            "Python": EEG_python,
            "MATLAB": EEG_matlab,
            "Octave": EEG_octave
        }

        for engine_name, eeg_result in all_results.items():
            with self.subTest(engine=f"{engine_name} shape check"):
                self.assertEqual(eeg_result['icaweights'].shape[1], n_chans)
                self.assertEqual(eeg_result['icasphere'].shape, (n_chans, n_chans))
                self.assertEqual(eeg_result['icaact'].shape[1], n_pnts)
                self.assertEqual(eeg_result['icaact'].shape[2], n_trials)

        print("All engines produced ICA fields with consistent shapes.")

        # Compare MATLAB and Octave results with tolerance
        # print("Comparing MATLAB and Octave results...")
        # np.testing.assert_allclose(EEG_matlab['icaweights'], EEG_octave['icaweights'],rtol=0.005, atol=1e-5,err_msg='MATLAB and Octave icaweights differ beyond tolerance')
        # np.testing.assert_allclose(EEG_matlab['icasphere'], EEG_octave['icasphere'],rtol=0.005, atol=1e-5,err_msg='MATLAB and Octave icasphere differ beyond tolerance')
        # np.testing.assert_allclose(EEG_matlab['icawinv'], EEG_octave['icawinv'],rtol=0.05, atol=0.0005,err_msg='MATLAB and Octave icawinv differ beyond tolerance')
        # # np.testing.assert_allclose(EEG_matlab['icaact'], EEG_octave['icaact'],rtol=0.005, atol=1e-5,err_msg='MATLAB and Octave icaact differ beyond tolerance')
        # print("MATLAB and Octave results are consistent.")

        # import sys
        # original_threshold = np.get_printoptions()['threshold']
        # np.set_printoptions(threshold=sys.maxsize)
        # print("pArray = np.", repr(EEG_python['icaweights']))
        # print("mArray = np.", repr(EEG_matlab['icaweights']))
        # np.set_printoptions(threshold=original_threshold)
        
        # plot the difference between each 2-D array and the difference between the 2-D arrays and save the figure
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im1 = axes[0].imshow(EEG_python['icaweights'], aspect='auto', cmap='viridis')
        axes[0].set_title('Python icaweights')
        fig.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(EEG_matlab['icaweights'], aspect='auto', cmap='viridis')
        axes[1].set_title('MATLAB icaweights')
        fig.colorbar(im2, ax=axes[1])
        
        diff = np.abs(EEG_python['icaweights'] - EEG_matlab['icaweights'])
        im3 = axes[2].imshow(diff, aspect='auto', cmap='magma')
        axes[2].set_title('Absolute Difference')
        fig.colorbar(im3, ax=axes[2])
        plt.savefig(os.path.join(local_url, 'icaweights_comparison.png'))
        plt.close()

        # save weights to MATLAB file
        import scipy.io
        scipy.io.savemat(os.path.join(local_url, 'icaweights_comparison.mat'), {'pArray': EEG_python['icaweights'], 'mArray': EEG_matlab['icaweights'], 'oArray': EEG_octave['icaweights']})

        # Compare Python and Octave results with tolerance
        print("Comparing Python and Matlab results...")
        print(repr(EEG_python['icasphere']))
        print(repr(EEG_matlab['icasphere']))
        np.testing.assert_allclose(EEG_python['icasphere'], EEG_matlab['icasphere'],rtol=0.005, atol=1e-5,err_msg='Python and Matlab icasphere differ beyond tolerance')
        np.testing.assert_allclose(EEG_python['icaweights'], EEG_matlab['icaweights'],rtol=0.005, atol=1e-5,err_msg='Python and Matlab icaweights differ beyond tolerance')
        np.testing.assert_allclose(EEG_python['icawinv'], EEG_matlab['icawinv'],rtol=0.05, atol=0.0005,err_msg='Python and Matlab icawinv differ beyond tolerance')
        # np.testing.assert_allclose(EEG_python['icaact'], EEG_octave['icaact'],rtol=0.005, atol=1e-5,err_msg='Python and Octave icaact differ beyond tolerance')
        print("Python and Octave results are consistent.")


if __name__ == '__main__':
    unittest.main()

# MATLAB code for manual comparison of the results
# EEG_python = pop_loadset('eeglab_data_picard_python.set');
# EEG_matlab = pop_loadset('eeglab_data_picard_matlab.set'); 
# EEG_octave = pop_loadset('eeglab_data_picard_octave.set');
#
# % Compare component activations
# eegplot(EEG_python.icaact, 'srate', EEG_python.srate, 'data2', EEG_matlab.icaact);
# title('Python (black) vs MATLAB (red) ICA activations');
#
# % Check correlation of weight matrices (they could be in different order and polarity)
# figure; imagesc(abs(corr(EEG_matlab.icaweights', EEG_python.icaweights'))); colorbar;
# title('Correlation of MATLAB vs Python ICA weights');
#
# % Compare MATLAB and Octave directly
# figure; hist(abs(EEG_octave.data(:) - EEG_matlab.data(:)), 100)
# title('Difference between MATLAB and Octave data');
#
# np.testing.assert_allclose(EEG_matlab['icaact'].flatten(), EEG_octave['icaact'].flatten(), 
#                                  rtol=1e-5, atol=1e-8,
#                                  err_msg='MATLAB and Octave results differ beyond tolerance') 