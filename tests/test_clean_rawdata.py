import os
import unittest
import psutil
from copy import deepcopy


import numpy as np

from eegprep import *
from eegprep.utils.testing import *


# where the test resources
web_root = 'https://sccntestdatasets.s3.us-east-2.amazonaws.com/'
local_url = os.path.join(os.path.dirname(__file__), '../data/')


def ensure_file(fname: str) -> str:
    """Download a file if it does not exist and return the local path."""
    full_url = f"{web_root}{fname}"
    local_file = f"{local_url}{fname}"
    if not os.path.exists(local_file):
        from urllib.request import urlretrieve
        urlretrieve(full_url, local_file)
    return local_file


class TestMATLABAccess(unittest.TestCase):

    def setUp(self):
        self.eeglab = eeglabcompat.get_eeglab('MAT')
        self.EEG = pop_loadset(ensure_file('FlankerTest.set'))

    def test_basic(self):
        result = self.eeglab.sqrt(4.0)
        self.assertEqual(result, 2.0, 'MATLAB sqrt() failed')

    def test_eeglab_presence(self):
        result = eeglabcompat.eeg_checkset(self.EEG, eeglab=self.eeglab)
        pass


class TestCleanFlatlines(unittest.TestCase):

    def setUp(self):
        # download file
        self.EEG = pop_loadset(ensure_file('FlankerTest.set'))
        self.EEG['data'][5, 1000:2000] = 3.5  # this should trigger
        self.EEG['data'][7, 2000:2100] = 4.5  # this should not (too short)
        self.EEG['data'][9, 3000:4000] = 5.5  # should trigger too
        self.EEG['data'][15, 5000:10000] = np.random.randn(5000)*1e-10  # should not trigger (too large amplitude)
        self.expected = self.EEG['data'][~np.isin(np.arange(self.EEG['nbchan']), [5, 9]), :]

    def test_clean_flatlines(self):
        cleaned_EEG = clean_flatlines(deepcopy(self.EEG), 3.5)
        np.testing.assert_equal(cleaned_EEG['data'], self.expected, err_msg='clean_flatlines() test failed')


class TestUtilFuncs(DebuggableTestCase):

    def setUp(self):
        self.eeglab = eeglabcompat.get_eeglab('MAT')

    def test_design_kaiser(self):
        from eegprep.utils import design_kaiser
        observed = design_kaiser(0.06, 0.08, 75, True)
        expected = np.asarray(self.eeglab.design_kaiser(0.06, 0.08, 75.0, True))
        np.testing.assert_almost_equal(observed.flatten(), expected.flatten(), 
                                       err_msg='design_kaiser() test failed')

    def test_design_fir_default_wnd(self):
        from eegprep.utils import design_fir
        observed = design_fir(234, [0.0, 0.06, 0.08, 1.0], [0, 0, 1, 1])
        expected = np.asarray(self.eeglab.design_fir(234.0, np.asarray([0.0, 0.06, 0.08, 1.0]), np.asarray([0.0, 0.0, 1.0, 1.0])))
        np.testing.assert_almost_equal(observed.flatten(), expected.flatten(),
                                       err_msg='test_design_fir_default_wnd() test failed')

    def test_design_fir_custom_wnd(self):
        from eegprep.utils import design_fir, design_kaiser
        wnd = design_kaiser(0.06, 0.08, 75.0, True)
        observed = design_fir(234, [0.0, 0.06, 0.08, 1.0], [0, 0, 1.0, 1.0], w=wnd)
        expected = np.asarray(self.eeglab.design_fir(234.0, np.asarray([0.0, 0.06, 0.08, 1.0]),
                                                     np.asarray([0, 0, 1.0, 1.0]), np.asarray([]), wnd))
        np.testing.assert_almost_equal(observed.flatten(), expected.flatten(),
                                       err_msg='test_design_fir_custom_wnd() test failed')

    def test_block_geometric_median(self):
        from eegprep.utils.stats import block_geometric_median
        np.random.seed(42)
        # generate heavy-tailed data with non-zero centroid and apply random rotation
        df = 3  # degrees of freedom for t-distribution
        center = np.arange(1, 33)  # non-zero centroid vector
        # random noise transform
        R = np.random.randn(32, 32)
        noise = np.random.standard_t(df, size=(5007, 32))
        X = noise.dot(R) + center
        observed = block_geometric_median(X, 10)
        expected = np.asarray(self.eeglab.block_geometric_median(X, 10.0))
        np.testing.assert_almost_equal(observed.flatten(), expected.flatten(),
                                       err_msg='block_geometric_median() test failed')
        
    def test_fit_eeg_distribution(self):
        from eegprep.utils.stats import fit_eeg_distribution
        from scipy.stats import genextreme
        x = genextreme.rvs(0.1, size=5007)
        observed, *_ = fit_eeg_distribution(x)  # returns 4 values, for now we check only the first
        expected = self.eeglab.fit_eeg_distribution(x)
        # compare numbers
        np.testing.assert_almost_equal(observed, expected,
                                       err_msg='fit_eeg_distribution() test failed')

class TestCleanDrifts(unittest.TestCase):

    def setUp(self):
        self.EEG = pop_loadset(ensure_file('FlankerTest.set'))

    def test_clean_drifts(self):
        eeglab = eeglabcompat.get_eeglab('MAT')

        # compare vs MATLAB
        expected = eeglab.clean_drifts(self.EEG, [3, 4], 75)
        cleaned1 = clean_drifts(deepcopy(self.EEG), [3, 4], 75, method='fir')
        compare_eeg(cleaned1['data'], expected['data'], 
                    err_msg='clean_drifts() failed')
        
        # compare FFT vs FIR
        cleaned2 = clean_drifts(deepcopy(self.EEG), [3, 4], 75, method='fft')
        compare_eeg(cleaned1['data'], cleaned2['data'], 
                    err_msg='clean_drifts() FFT mode test failed',atol=2e-7)
        

class TestCleanChannels(unittest.TestCase):

    def setUp(self):
        self.EEG = pop_loadset(ensure_file('EmotionValence.set'))

    def test_clean_channels_nolocs(self):
        eeglab = eeglabcompat.get_eeglab('MAT')
        cleaned, _ = clean_channels_nolocs(deepcopy(self.EEG), 0.9)
        expected = eeglab.clean_channels_nolocs(self.EEG, 0.9)
        compare_eeg(cleaned['data'], expected['data'],
                    err_msg='clean_channels_nolocs() failed')

    def test_clean_channels_locs(self):
        cleaned = clean_channels(deepcopy(self.EEG), 0.9)
        eeglab = eeglabcompat.get_eeglab('MAT')
        expected = eeglab.clean_channels(self.EEG, 0.9)
        compare_eeg(cleaned['data'], expected['data'],
                    err_msg='clean_channels() failed')


class TestCleanASR(unittest.TestCase):

    def setUp(self):
        self.EEG = pop_loadset(ensure_file('EmotionValence.set'))

    def test_clean_asr_nowindow(self):
        cleaned = clean_asr(deepcopy(self.EEG), ref_maxbadchannels='off')
        eeglab = eeglabcompat.get_eeglab('MAT')
        expected = eeglab.clean_asr(self.EEG, [],[],[],[], 'off')
        compare_eeg(cleaned['data'], expected['data'], 
                    atol=0, rtol=1e-6, # because of eigh() precision differences
                    err_msg='clean_asr() failed vs MATLAB')


class TestCleanWindows(unittest.TestCase):

    def setUp(self):
        self.EEG = pop_loadset(ensure_file('EmotionValence.set'))

        # ------------------------------------------------------------------
        # Inject synthetic high‑amplitude artefacts so that clean_windows()
        # has something to remove. We draw 20 random stretches (200‑2000
        # samples) and corrupt a random subset of channels in each stretch
        # by adding scaled Gaussian noise.
        # ------------------------------------------------------------------
        rng = np.random.default_rng(seed=42)  # deterministic for the test

        n_stretches = 20
        n_chans = self.EEG['nbchan']
        n_samp = self.EEG['pnts']

        for _ in range(n_stretches):
            # Random stretch length between 200 and 2000 samples (inclusive)
            stretch_len = int(rng.integers(200, 2001))

            # Random start index ensuring we stay within bounds
            start = int(rng.integers(0, max(1, n_samp - stretch_len)))
            stop = start + stretch_len

            # Random number of channels (at least 1, at most all channels)
            n_corrupt = int(rng.integers(1, n_chans + 1))
            chan_idx = rng.choice(n_chans, size=n_corrupt, replace=False)

            # Local scale (robust): std of the windowed data
            window_data = self.EEG['data'][chan_idx, start:stop]
            local_std = float(np.std(window_data, ddof=0))
            if local_std == 0:
                local_std = 1.0  # avoid degenerate case

            # Heavy‑tailed positive scale factor (Gamma with mean ≈3)
            scale_factor = float(rng.gamma(shape=2.0, scale=1.5))

            noise = rng.standard_normal(size=(n_corrupt, stretch_len)) * local_std * scale_factor
            self.EEG['data'][chan_idx, start:stop] += noise

    def test_clean_windows(self):
        cleaned, _ = clean_windows(deepcopy(self.EEG))
        eeglab = eeglabcompat.get_eeglab('MAT')
        expected = eeglab.clean_windows(self.EEG)
        compare_eeg(cleaned['data'], expected['data'], 
                    err_msg='clean_windows() failed vs MATLAB')


# ------------------------------------------------------------------------------
#                               clean_artifacts
# ------------------------------------------------------------------------------


class TestCleanArtifacts(DebuggableTestCase):

    def setUp(self):
        # Use the same dataset as other heavy‑duty tests
        self.EEG = pop_loadset(ensure_file('EmotionValence.set'))

    def test_clean_artifacts_defaults(self):
        """Compare Python clean_artifacts against MATLAB implementation (default params).
        """
        # --- Python version ---
        cleaned_py, _, _, _ = clean_artifacts(deepcopy(self.EEG))

        # --- MATLAB reference ---
        eeglab = eeglabcompat.get_eeglab('MAT')
        # Call with the matching name‑value pair
        expected_mat = eeglab.clean_artifacts(self.EEG)

        compare_eeg(
            cleaned_py['data'],
            expected_mat['data'],
            rtol=0,  
            atol=1e-5,  # limit to 1e-5 uV likely due to solver differences
            err_msg='clean_artifacts() failed vs MATLAB'
        )


if __name__ == "__main__":
    TestCleanArtifacts.debugTestCase()
    # TestUtilFuncs.debugTestCase()
    unittest.main()
    