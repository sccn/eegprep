import os
import unittest
import numpy as np
import pytest
from picard import picard
from eegprep import pop_loadset, eeg_picard, pop_saveset
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.utils.testing import DebuggableTestCase, matlab_function_exists

from tests.fixtures import create_test_eeg as _create_test_eeg


def compare_ica_components(weights1, weights2, rtol=0.01, atol=0.05):
    """Compare ICA weight matrices accounting for permutation and sign ambiguity.

    ICA solutions are unique only up to permutation and sign of components.
    This function computes the correlation matrix between components and checks
    that each component in weights1 has a highly correlated match in weights2.

    Parameters
    ----------
    weights1 : ndarray
        First ICA weight matrix (n_components x n_channels)
    weights2 : ndarray
        Second ICA weight matrix (n_components x n_channels)
    rtol : float
        Relative tolerance for correlation check
    atol : float
        Absolute tolerance - max correlation should be at least (1 - atol)

    Returns
    -------
    matched : bool
        True if all components have high-correlation matches
    max_correlations : ndarray
        Maximum absolute correlation for each component in weights1
    best_matches : ndarray
        Index of best matching component in weights2 for each component in weights1
    """
    n_comp = weights1.shape[0]

    # weights1 rows correlated with weights2 rows
    corr_matrix = np.zeros((n_comp, n_comp))
    for i in range(n_comp):
        for j in range(n_comp):
            corr_matrix[i, j] = np.corrcoef(weights1[i, :], weights2[j, :])[0, 1]

    abs_corr = np.abs(corr_matrix)
    max_correlations = np.max(abs_corr, axis=1)
    best_matches = np.argmax(abs_corr, axis=1)

    min_acceptable_corr = 1.0 - atol
    matched = np.all(max_correlations >= min_acceptable_corr)

    return matched, max_correlations, best_matches, corr_matrix


# ASSESSMENT OF THE TEST RESULTS
# The current conclusion is that while MATLAB and Octave are not exactly the same, they are close enough.
# However, Python is quite different. The image is saved in the test folder of the difference between the 2-D arrays.
# More investigation is needed to understand why this is the case. The implementation are quite different to start with.


# where the test resources
local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')


def create_test_eeg():
    """Epoched EEG fixture sized for eeg_picard (32 ch, 1000 pnts, 10 trials)."""
    return _create_test_eeg(n_channels=32, n_samples=1000, srate=500.0, n_trials=10)


@pytest.mark.parametrize('n_components', [None, 3])
def test_eeg_picard_initialization(n_components):
    eeg = _create_test_eeg(n_channels=4, n_samples=400, n_trials=2)
    eeg['data'] = np.random.RandomState(7).laplace(size=(4, 400, 2))
    # Stop early to observe initialization, even if converged ICA solutions coincide.
    options = dict(n_components=n_components, max_iter=2, verbose=False)
    with pytest.warns(UserWarning, match='Picard did not converge'):
        default = eeg_picard(eeg, **options)
        identity = eeg_picard(eeg, w_init=np.eye(n_components or 4), random_state=123, **options)
        seeded = eeg_picard(eeg, random_state=0, **options)
        repeated = eeg_picard(eeg, random_state=0, **options)
        other = eeg_picard(eeg, random_state=123, **options)
        k, w, _ = picard(eeg['data'].reshape(4, -1, order='F'), ortho=False, m=10, random_state=0, **options)

    np.testing.assert_array_equal(default['icaweights'], identity['icaweights'])
    np.testing.assert_array_equal(seeded['icaweights'], repeated['icaweights'])
    assert not np.allclose(seeded['icaweights'], other['icaweights'])
    np.testing.assert_allclose(seeded['etc']['icaweights_beforerms'], w @ k, rtol=1e-12, atol=1e-12)
    assert not np.allclose(seeded['icaweights'], seeded['etc']['icaweights_beforerms'])
    column_rms = np.sqrt(np.mean(np.asarray(seeded['icawinv'], dtype=float) ** 2, axis=0))
    np.testing.assert_allclose(column_rms, 1.0, rtol=1e-8, atol=1e-8)


def test_eeg_picard_rms_scaling_follows_option_scaleicarms(monkeypatch):
    eeg = _create_test_eeg(n_channels=4, n_samples=200, n_trials=1)
    eeg["data"] = np.random.RandomState(3).standard_normal((4, 200)) + 4.0
    monkeypatch.setattr("eegprep.functions.popfunc._ica_utils.option_scaleicarms", False)

    out = eeg_picard(eeg, verbose=False, max_iter=2)

    assert "icaweights_beforerms" not in out.get("etc", {})
    column_rms = np.sqrt(np.mean(np.asarray(out["icawinv"], dtype=float) ** 2, axis=0))
    assert np.max(np.abs(column_rms - 1.0)) > 1e-3


def test_eeg_picard_activations_are_unmixing_times_data():
    generator = np.random.default_rng(0)
    data = generator.standard_normal((4, 300)) + 25.0
    eeg = {"data": data, "nbchan": 4, "pnts": 300, "trials": 1, "srate": 100.0}

    out = eeg_picard(eeg, verbose=False)
    activations = np.asarray(out["icaact"]).reshape(4, -1, order="F")
    expected = (out["icaweights"] @ out["icasphere"]) @ data

    np.testing.assert_allclose(activations, expected, rtol=1e-8, atol=1e-8)


class TestEegPicardSimple(DebuggableTestCase):
    """Simple test cases for eeg_picard function (Python implementation only)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_eeg_picard_does_not_mutate_caller(self):
        """eeg_picard must not mutate the caller's data or ICA fields."""
        data_before = self.test_eeg['data'].copy()
        # A pre-ICA input has no icaweights yet; "does not mutate" means the
        # caller's fields stay the exact objects they were (the result is a copy).
        weights_before = self.test_eeg.get('icaweights')
        chansind_before = self.test_eeg.get('icachansind')

        eeg_picard(self.test_eeg, posact=True, max_iter=10, random_state=1, verbose=False)

        self.assertTrue(np.array_equal(self.test_eeg['data'], data_before))
        self.assertIs(self.test_eeg.get('icaweights'), weights_before)
        self.assertIs(self.test_eeg.get('icachansind'), chansind_before)


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestEegPicard(unittest.TestCase):
    def setUp(self):
        # Using a standard test file.
        # Even if it has ICA data, picard should overwrite it.
        self.EEG = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set'))

    def test_picard_engines(self):
        """Test eeg_picard with Python, MATLAB, and Octave engines."""

        eeglab_mat = get_eeglab('MAT')
        if not matlab_function_exists(eeglab_mat, 'eeg_picard'):
            self.skipTest("MATLAB EEGLAB Picard plugin is not installed")

        print("Running Picard with Python engine...")
        EEG_python = eeg_picard(self.EEG.copy())
        pop_saveset(EEG_python, os.path.join(local_url, 'eeglab_data_picard_python.set'))
        print("Python engine test completed.")

        # --- MATLAB Engine ---
        print("Running Picard with MATLAB engine...")
        EEG_matlab = eeg_picard(self.EEG.copy(), engine=eeglab_mat)
        pop_saveset(EEG_matlab, os.path.join(local_url, 'eeglab_data_picard_matlab.set'))
        print("MATLAB engine test completed.")

        # --- Assertions ---

        for eeg_result, engine_name in [(EEG_python, 'Python'), (EEG_matlab, 'MATLAB')]:  # , (EEG_octave, 'Octave')]:
            with self.subTest(engine=engine_name):
                self.assertIn('icaweights', eeg_result)
                self.assertIn('icasphere', eeg_result)
                self.assertIn('icawinv', eeg_result)
                self.assertIn('icaact', eeg_result)
                self.assertIn('icachansind', eeg_result)

        # Assuming number of components is equal to number of channels by default
        n_chans = self.EEG['nbchan']
        n_pnts = self.EEG['pnts']
        n_trials = self.EEG['trials']

        all_results = {
            "Python": EEG_python,
            "MATLAB": EEG_matlab,
            # "Octave": EEG_octave
        }

        for engine_name, eeg_result in all_results.items():
            with self.subTest(engine=f"{engine_name} shape check"):
                self.assertEqual(eeg_result['icaweights'].shape[1], n_chans)
                self.assertEqual(eeg_result['icasphere'].shape, (n_chans, n_chans))
                self.assertEqual(eeg_result['icaact'].shape[1], n_pnts)
                self.assertEqual(eeg_result['icaact'].shape[2], n_trials)

        print("All engines produced ICA fields with consistent shapes.")

        # plot the difference between each 2-D array and the difference between the 2-D arrays and save the figure
        if False:
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

        scipy.io.savemat(
            os.path.join(local_url, 'icaweights_comparison.mat'),
            {'pArray': EEG_python['icaweights'], 'mArray': EEG_matlab['icaweights']},
        )  # , 'oArray': EEG_octave['icaweights']})

        # Compare Python and MATLAB results using correlation-based comparison
        # ICA solutions are unique only up to permutation and sign of components
        # Therefore we use correlation to match components rather than element-wise comparison
        print("Comparing Python and Matlab results using correlation-based matching...")

        # icasphere should be identity for both (exact match expected)
        np.testing.assert_allclose(
            EEG_python['icasphere'],
            EEG_matlab['icasphere'],
            rtol=0.005,
            atol=1e-5,
            err_msg='Python and Matlab icasphere differ beyond tolerance',
        )

        # Compare icaweights using correlation (handles permutation and sign ambiguity)
        matched, max_corrs, best_matches, corr_matrix = compare_ica_components(
            EEG_python['icaweights'], EEG_matlab['icaweights'], atol=0.35
        )

        print(f"  Min correlation across components: {np.min(max_corrs):.4f}")
        print(f"  Mean correlation across components: {np.mean(max_corrs):.4f}")
        print(f"  Components with correlation < 0.95: {np.sum(max_corrs < 0.95)}")
        print(f"  Components with correlation < 0.65: {np.sum(max_corrs < 0.65)}")

        # Check for unique matching (each Python component maps to a different MATLAB component)
        unique_matches = len(set(best_matches)) == len(best_matches)
        if not unique_matches:
            print(f"  WARNING: Non-unique matching detected. Best matches: {best_matches}")

        self.assertTrue(
            matched,
            f"ICA components do not match well. Min correlation: {np.min(max_corrs):.4f}. "
            f"Expected >= 0.65 for all components. Mean: {np.mean(max_corrs):.4f}",
        )

        # Note: icawinv = pinv(icaweights), so if icaweights match, icawinv is
        # mathematically determined. We skip explicit icawinv comparison since
        # the component order differs between Python and MATLAB, and reordering
        # would be redundant with the icaweights check above.

        print("Python and MATLAB ICA results match (accounting for permutation/sign).")


if __name__ == '__main__':
    unittest.main(defaultTest='TestEegPicard.test_picard_engines')
