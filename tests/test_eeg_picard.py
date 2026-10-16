import os
import unittest
import warnings
from contextlib import contextmanager

import numpy as np
import pytest
from picard import picard

from eegprep import eeg_picard, pop_loadset, pop_saveset
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.utils.testing import DebuggableTestCase, matlab_function_exists
from tests.eeg_options_snapshot import eeg_options_snapshot, restore_eeg_options
from tests.fixtures import create_test_eeg as _create_test_eeg
from tests.test_ds003061_stage_parity import (
    PICARD_AMARI_MAX,
    PICARD_CORR_MIN,
    amari_distance,
    matched_mixing_correlation,
)


@contextmanager
def _matlab_double_precision(engine):
    """Turn ``option_single`` off and write the snapshotted options file back."""
    options_path, original = eeg_options_snapshot(engine)
    try:
        engine.eval("pop_editoptions('option_single', 0);", nargout=0)
        engine.eval("eeglab_options;", nargout=0)
        yield
    finally:
        restore_eeg_options(options_path, original)
        engine.eval("clear functions", nargout=0)
        engine.eval("eeglab_options;", nargout=0)


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
        """Compare Python and MATLAB Picard on the sample set in double precision.

        MATLAB ``eeg_picard`` must call ``picard_python_port`` (identity
        initialization). ``option_single`` is turned off for the call and the
        options file is restored afterward.
        """

        eeglab_mat = get_eeglab('MAT')
        engine = eeglab_mat.engine
        if not matlab_function_exists(eeglab_mat, 'eeg_picard'):
            self.skipTest("MATLAB EEGLAB Picard plugin is not installed")
        source = str(engine.eval("fileread(which('eeg_picard'))", nargout=1))
        if "picard_python_port" not in source:
            self.fail(
                "MATLAB eeg_picard does not call picard_python_port. Install mind-inria/picard "
                "617ce6ae669cfbd1dfedba6a5fce39a602b99ac3. The EEGLAB PICARD 2.0 extension uses a seeded random start."
            )

        print("Running Picard with Python engine...")
        EEG_python = eeg_picard(self.EEG.copy())
        pop_saveset(EEG_python, os.path.join(local_url, 'eeglab_data_picard_python.set'))
        print("Python engine test completed.")

        with _matlab_double_precision(engine):
            option_single = int(engine.eval("option_single", nargout=1))
            self.assertEqual(option_single, 0)
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

        np.testing.assert_allclose(
            EEG_python['icasphere'],
            EEG_matlab['icasphere'],
            rtol=0.005,
            atol=1e-5,
            err_msg='Python and Matlab icasphere differ beyond tolerance',
        )

        python_unmixing = np.asarray(EEG_python['icaweights'], dtype=float) @ np.asarray(
            EEG_python['icasphere'], dtype=float
        )
        distance = amari_distance(python_unmixing, np.asarray(EEG_matlab['icawinv'], dtype=float))
        mean_correlation, min_correlation = matched_mixing_correlation(EEG_python['icawinv'], EEG_matlab['icawinv'])
        parity = (
            f"picard sample parity: amari={distance:.6e} "
            f"corr_mean={mean_correlation:.6f} corr_min={min_correlation:.6f}"
        )
        print(parity)
        warnings.warn(parity, UserWarning, stacklevel=1)
        self.assertLessEqual(distance, PICARD_AMARI_MAX, f"AMARI {distance:.6e} exceeds {PICARD_AMARI_MAX:.6e}")
        self.assertGreaterEqual(
            min_correlation,
            PICARD_CORR_MIN,
            f"min matched correlation {min_correlation:.6f} is below {PICARD_CORR_MIN}",
        )


if __name__ == '__main__':
    unittest.main(defaultTest='TestEegPicard.test_picard_engines')
