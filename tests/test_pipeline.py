import os
import unittest
import numpy as np
from eegprep import pop_loadset, clean_artifacts, eeg_picard, iclabel
from eegprep.eeglabcompat import get_eeglab
from eegprep.eeg_compare import eeg_compare
from eegprep.utils.testing import compare_eeg

@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
def test_pipeline():
    """Test pipeline: clean_artifacts -> eeg_picard -> iclabel, comparing Python and MATLAB at each step."""
    # where the test resources
    local_url = os.path.join(os.path.dirname(__file__), '../data/')
    fname = os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')
    EEG = pop_loadset(fname)

    # --- Step 1: clean_artifacts (channel cleaning only) ---
    # Channel cleaning only: BurstCriterion='off'
    EEG_py_ch, *_ = clean_artifacts(EEG, BurstCriterion='off', ChannelCriterion=0.8)
    eeglab = get_eeglab('MAT')
    EEG_mat_ch = eeglab.clean_artifacts(EEG, 'BurstCriterion', 'off', 'ChannelCriterion', 0.8)
    # eeg_compare(EEG_py_ch, EEG_mat_ch)
    compare_eeg(EEG_py_ch['data'], EEG_mat_ch['data'], rtol=0.005, atol=1e-5, err_msg='clean_artifacts() channel cleaning Python vs MATLAB failed')
    print("clean_artifacts() channel cleaning Python vs MATLAB passed\n\n\n")

    # --- Step 1b: clean_artifacts (burst cleaning only, after channel cleaning) ---
    # Burst cleaning only: ChannelCriterion='off'
    EEG_py = clean_artifacts(EEG_py_ch, ChannelCriterion='off')
    EEG_mat = eeglab.clean_artifacts(EEG_mat_ch, ChannelCriterion='off', BurstCriterion=5.0)
    eeg_compare(EEG_py, EEG_mat)
    compare_eeg(EEG_py['data'], EEG_mat['data'], rtol=0.005, atol=1e-5, err_msg='clean_artifacts() burst cleaning Python vs MATLAB failed')
    print("clean_artifacts() burst cleaning Python vs MATLAB passed\n\n\n")
    
    # --- Step 2: eeg_picard ---
    EEG_py_ica = eeg_picard(EEG_py)
    EEG_mat_ica = eeg_picard(EEG_mat, engine=eeglab)
    # Compare ICA fields
    for field in ['icaweights', 'icasphere', 'icawinv', 'icaact', 'icachansind']:
        assert field in EEG_py_ica and field in EEG_mat_ica, f"Missing ICA field: {field}"
    eeg_compare(EEG_py_ica['icaweights'], EEG_mat_ica['icaweights'])
    eeg_compare(EEG_py_ica['icasphere'], EEG_mat_ica['icasphere'])
    eeg_compare(EEG_py_ica['icawinv'], EEG_mat_ica['icawinv'])
    print("eeg_picard() Python vs MATLAB passed\n\n\n")
    
    # --- Step 3: iclabel ---
    EEG_py_lbl = iclabel(EEG_py_ica)
    EEG_mat_lbl = iclabel(EEG_mat_ica, engine='matlab')
    # Check ICLabel output structure
    for EEG_lbl in [EEG_py_lbl, EEG_mat_lbl]:
        assert 'etc' in EEG_lbl and 'ic_classification' in EEG_lbl['etc'] and 'ICLabel' in EEG_lbl['etc']['ic_classification'], 'ICLabel output missing'
    res_py = EEG_py_lbl['etc']['ic_classification']['ICLabel']['classifications'].flatten()
    res_mat = EEG_mat_lbl['etc']['ic_classification']['ICLabel']['classifications'].flatten()
    eeg_compare(res_py, res_mat)

if __name__ == "__main__":
    test_pipeline()
    print("Pipeline test passed.") 