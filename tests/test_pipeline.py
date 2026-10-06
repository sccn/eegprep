"""
Test suite for pipeline: clean_artifacts -> eeg_picard -> iclabel
"""

import os
import unittest
from copy import deepcopy
from eegprep import pop_loadset, clean_artifacts
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.eeg_compare import eeg_compare
from eegprep.utils.testing import (
    compare_eeg,
    DebuggableTestCase,
)


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestPipeline(DebuggableTestCase):
    """Test pipeline: clean_artifacts -> eeg_picard -> iclabel, comparing Python and MATLAB at each step."""

    def setUp(self):
        """Set up test fixtures."""
        local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')
        fname = os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')
        self.EEG = pop_loadset(fname)
        self.eeglab = get_eeglab('MAT')

    def test_clean_artifacts_burst_cleaning(self):
        """Test clean_artifacts burst cleaning step (ChannelCriterion='off')."""
        # First do channel cleaning
        EEG_py_ch, *_ = clean_artifacts(deepcopy(self.EEG), BurstCriterion='off', ChannelCriterion=0.8)
        EEG_mat_ch = self.eeglab.clean_artifacts(deepcopy(self.EEG), 'BurstCriterion', 'off', 'ChannelCriterion', 0.8)

        # Then do burst cleaning only: ChannelCriterion='off'
        EEG_py, *_ = clean_artifacts(EEG_py_ch, ChannelCriterion='off')
        EEG_mat = self.eeglab.clean_artifacts(EEG_mat_ch, 'ChannelCriterion', 'off', 'BurstCriterion', 5.0)

        print("\n" + "=" * 80)
        print("Step 1b: clean_artifacts (burst cleaning only)")
        print("=" * 80)
        eeg_summary = eeg_compare(EEG_py, EEG_mat)
        print(f"\n{eeg_summary}")
        data_summary = compare_eeg(
            EEG_py['data'],
            EEG_mat['data'],
            rtol=0.005,
            atol=1e-5,
            err_msg='clean_artifacts() burst cleaning Python vs MATLAB failed',
        )
        print(f"\n{data_summary}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    unittest.main()
