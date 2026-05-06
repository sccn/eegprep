"""
Test suite for pipeline: clean_artifacts -> eeg_picard -> iclabel
"""
import os
import re
import unittest
import numpy as np
from copy import deepcopy
from eegprep import pop_loadset, clean_artifacts, eeg_picard, iclabel
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.eeg_compare import eeg_compare
from eegprep.utils.testing import (
    compare_eeg,
    DebuggableTestCase,
    has_optional_dependency,
    matlab_function_exists,
)

@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
def test_pipeline():
    """Test pipeline: clean_artifacts -> eeg_picard -> iclabel, comparing Python and MATLAB at each step."""
    # where the test resources
    local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')
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

@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestPipeline(DebuggableTestCase):
    """Test pipeline: clean_artifacts -> eeg_picard -> iclabel, comparing Python and MATLAB at each step."""

    def setUp(self):
        """Set up test fixtures."""
        local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')
        fname = os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')
        self.EEG = pop_loadset(fname)
        self.eeglab = get_eeglab('MAT')
        self.has_matlab_picard = matlab_function_exists(self.eeglab, 'eeg_picard')

    def test_clean_artifacts_channel_cleaning(self):
        """Test clean_artifacts channel cleaning step (BurstCriterion='off')."""
        # Channel cleaning only: BurstCriterion='off'
        # Use deepcopy to ensure Python doesn't modify the original EEG
        EEG_py_ch, *_ = clean_artifacts(deepcopy(self.EEG), BurstCriterion='off', ChannelCriterion=0.8)
        # MATLAB also needs a fresh copy since it may modify the EEG structure
        EEG_mat_ch = self.eeglab.clean_artifacts(deepcopy(self.EEG), 'BurstCriterion', 'off', 'ChannelCriterion', 0.8)

        print("\n" + "="*80)
        print("Step 1: clean_artifacts (channel cleaning only)")
        print("="*80)
        summary = compare_eeg(
            EEG_py_ch['data'],
            EEG_mat_ch['data'],
            rtol=0.005,
            atol=1e-5,
            err_msg='clean_artifacts() channel cleaning Python vs MATLAB failed'
        )
        print(summary)
        print("="*80 + "\n")

    def test_clean_artifacts_burst_cleaning(self):
        """Test clean_artifacts burst cleaning step (ChannelCriterion='off')."""
        # First do channel cleaning
        EEG_py_ch, *_ = clean_artifacts(deepcopy(self.EEG), BurstCriterion='off', ChannelCriterion=0.8)
        EEG_mat_ch = self.eeglab.clean_artifacts(deepcopy(self.EEG), 'BurstCriterion', 'off', 'ChannelCriterion', 0.8)

        # Then do burst cleaning only: ChannelCriterion='off'
        EEG_py, *_ = clean_artifacts(EEG_py_ch, ChannelCriterion='off')
        EEG_mat = self.eeglab.clean_artifacts(EEG_mat_ch, 'ChannelCriterion', 'off', 'BurstCriterion', 5.0)

        print("\n" + "="*80)
        print("Step 1b: clean_artifacts (burst cleaning only)")
        print("="*80)
        eeg_summary = eeg_compare(EEG_py, EEG_mat)
        print(f"\n{eeg_summary}")
        data_summary = compare_eeg(
            EEG_py['data'],
            EEG_mat['data'],
            rtol=0.005,
            atol=1e-5,
            err_msg='clean_artifacts() burst cleaning Python vs MATLAB failed'
        )
        print(f"\n{data_summary}")
        print("="*80 + "\n")

    def test_eeg_picard(self):
        """Test eeg_picard ICA decomposition."""
        if not self.has_matlab_picard:
            self.skipTest("MATLAB EEGLAB Picard plugin is not installed")

        # Prepare data: channel cleaning + burst cleaning
        EEG_py_ch, *_ = clean_artifacts(deepcopy(self.EEG), BurstCriterion='off', ChannelCriterion=0.8)
        EEG_mat_ch = self.eeglab.clean_artifacts(deepcopy(self.EEG), 'BurstCriterion', 'off', 'ChannelCriterion', 0.8)
        EEG_py, *_ = clean_artifacts(EEG_py_ch, ChannelCriterion='off')
        EEG_mat = self.eeglab.clean_artifacts(EEG_mat_ch, 'ChannelCriterion', 'off', 'BurstCriterion', 5.0)

        # Run ICA
        EEG_py_ica = eeg_picard(EEG_py)
        EEG_mat_ica = eeg_picard(EEG_mat, engine=self.eeglab)

        # Compare ICA fields
        print("\n" + "="*80)
        print("Step 2: eeg_picard (ICA decomposition)")
        print("="*80)
        for field in ['icaweights', 'icasphere', 'icawinv', 'icaact', 'icachansind']:
            self.assertIn(field, EEG_py_ica, f"Missing ICA field in Python: {field}")
            self.assertIn(field, EEG_mat_ica, f"Missing ICA field in MATLAB: {field}")

        print("\nComparing icaweights:")
        weights_summary = eeg_compare(EEG_py_ica['icaweights'], EEG_mat_ica['icaweights'])
        print(weights_summary)

        print("\nComparing icasphere:")
        sphere_summary = eeg_compare(EEG_py_ica['icasphere'], EEG_mat_ica['icasphere'])
        print(sphere_summary)

        print("\nComparing icawinv:")
        winv_summary = eeg_compare(EEG_py_ica['icawinv'], EEG_mat_ica['icawinv'])
        print(winv_summary)
        print("="*80 + "\n")

    def test_iclabel(self):
        """Test iclabel component classification."""
        if not self.has_matlab_picard:
            self.skipTest("MATLAB EEGLAB Picard plugin is not installed")
        if not has_optional_dependency('torch'):
            self.skipTest("PyTorch is not installed; install eegprep[torch] to run ICLabel parity")

        # Prepare data: channel cleaning + burst cleaning + ICA
        EEG_py_ch, *_ = clean_artifacts(deepcopy(self.EEG), BurstCriterion='off', ChannelCriterion=0.8)
        EEG_mat_ch = self.eeglab.clean_artifacts(deepcopy(self.EEG), 'BurstCriterion', 'off', 'ChannelCriterion', 0.8)
        EEG_py, *_ = clean_artifacts(EEG_py_ch, ChannelCriterion='off')
        EEG_mat = self.eeglab.clean_artifacts(EEG_mat_ch, 'ChannelCriterion', 'off', 'BurstCriterion', 5.0)
        EEG_py_ica = eeg_picard(EEG_py)
        EEG_mat_ica = eeg_picard(EEG_mat, engine=self.eeglab)

        # Run ICLabel
        EEG_py_lbl = iclabel(EEG_py_ica)
        EEG_mat_lbl = iclabel(EEG_mat_ica, engine='matlab')

        # Check ICLabel output structure
        print("\n" + "="*80)
        print("Step 3: iclabel (component classification)")
        print("="*80)
        for EEG_lbl in [EEG_py_lbl, EEG_mat_lbl]:
            self.assertIn('etc', EEG_lbl, 'Missing etc field')
            self.assertIn('ic_classification', EEG_lbl['etc'], 'Missing ic_classification field')
            self.assertIn('ICLabel', EEG_lbl['etc']['ic_classification'], 'Missing ICLabel field')

        res_py = EEG_py_lbl['etc']['ic_classification']['ICLabel']['classifications'].flatten()
        res_mat = EEG_mat_lbl['etc']['ic_classification']['ICLabel']['classifications'].flatten()
        print("\nComparing ICLabel classifications:")
        iclabel_summary = eeg_compare(res_py, res_mat)
        print(iclabel_summary)
        print("="*80 + "\n")

    def test_z_full_pipeline(self):
        """Test the complete pipeline end-to-end."""
        if not self.has_matlab_picard:
            self.skipTest("MATLAB EEGLAB Picard plugin is not installed")
        if not has_optional_dependency('torch'):
            self.skipTest("PyTorch is not installed; install eegprep[torch] to run full pipeline parity")

        print("\n" + "="*80)
        print("Full Pipeline Test: clean_artifacts -> eeg_picard -> iclabel")
        print("="*80)

        # Run the pipeline once and collect all eeg_compare summaries
        summaries = {}

        # Step 1: Channel cleaning
        EEG_py_ch, *_ = clean_artifacts(deepcopy(self.EEG), BurstCriterion='off', ChannelCriterion=0.8)
        EEG_mat_ch = self.eeglab.clean_artifacts(deepcopy(self.EEG), 'BurstCriterion', 'off', 'ChannelCriterion', 0.8)
        data_summary_1 = compare_eeg(EEG_py_ch['data'], EEG_mat_ch['data'], rtol=0.005, atol=1e-5,
                                     err_msg='clean_artifacts() channel cleaning Python vs MATLAB failed')
        print(f"\nStep 1 - Channel cleaning data comparison:\n{data_summary_1}")

        # Step 1b: Burst cleaning
        EEG_py, *_ = clean_artifacts(EEG_py_ch, ChannelCriterion='off')
        EEG_mat = self.eeglab.clean_artifacts(EEG_mat_ch, 'ChannelCriterion', 'off', 'BurstCriterion', 5.0)
        summaries['burst_cleaning_eeg'] = eeg_compare(EEG_py, EEG_mat)
        data_summary_1b = compare_eeg(EEG_py['data'], EEG_mat['data'], rtol=0.005, atol=1e-5,
                                      err_msg='clean_artifacts() burst cleaning Python vs MATLAB failed')
        print(f"\nStep 1b - Burst cleaning data comparison:\n{data_summary_1b}")

        # Step 2: ICA
        EEG_py_ica = eeg_picard(EEG_py)
        EEG_mat_ica = eeg_picard(EEG_mat, engine=self.eeglab)
        summaries['icaweights'] = eeg_compare(EEG_py_ica['icaweights'], EEG_mat_ica['icaweights'])
        summaries['icasphere'] = eeg_compare(EEG_py_ica['icasphere'], EEG_mat_ica['icasphere'])
        summaries['icawinv'] = eeg_compare(EEG_py_ica['icawinv'], EEG_mat_ica['icawinv'])

        # Step 3: ICLabel
        EEG_py_lbl = iclabel(EEG_py_ica)
        EEG_mat_lbl = iclabel(EEG_mat_ica, engine='matlab')
        res_py = EEG_py_lbl['etc']['ic_classification']['ICLabel']['classifications'].flatten()
        res_mat = EEG_mat_lbl['etc']['ic_classification']['ICLabel']['classifications'].flatten()
        summaries['iclabel_classifications'] = eeg_compare(res_py, res_mat)

        # Print consolidated eeg_compare summary as a table
        print("\n" + "="*80)
        print("Full Pipeline Test - Consolidated eeg_compare Summary Table")
        print("="*80)

        # Helper function to extract metrics from summary string
        def extract_metrics(summary_str):
            """Extract key metrics from summary string."""
            metrics = {
                'max_abs_diff': 'N/A',
                'mean_abs_diff': 'N/A',
                'rms_diff': 'N/A',
                'max_rel_diff': 'N/A',
                'mismatch_pct': 'N/A'
            }
            if 'Array Comparison Summary' in summary_str:
                for line in summary_str.split('\n'):
                    if 'Max absolute difference' in line:
                        metrics['max_abs_diff'] = line.split(':')[1].strip()
                    elif 'Mean absolute difference' in line:
                        metrics['mean_abs_diff'] = line.split(':')[1].strip()
                    elif 'RMS difference' in line:
                        metrics['rms_diff'] = line.split(':')[1].strip()
                    elif 'Max relative difference' in line:
                        metrics['max_rel_diff'] = line.split(':')[1].strip()
                    elif 'Mismatched elements' in line and '%' in line:
                        # Extract percentage like "Mismatched elements (> 1e-10): 900 (100.00%)"
                        # Find the last occurrence of (X.XX%) pattern
                        match = re.search(r'\(([\d.]+)%\)', line)
                        if match:
                            metrics['mismatch_pct'] = match.group(1) + '%'
                        else:
                            metrics['mismatch_pct'] = 'N/A'
            elif 'Found' in summary_str and 'differences' in summary_str:
                # For EEG structure comparisons
                metrics['max_abs_diff'] = 'See details'
                metrics['mismatch_pct'] = summary_str.split('Found')[1].split('total')[0].strip() + ' diff'
            return metrics

        # Organize summaries by step
        step_data = []

        # Step 1: clean_artifacts (use burst_cleaning_eeg as it's the final output)
        if 'burst_cleaning_eeg' in summaries:
            step1_summary = summaries['burst_cleaning_eeg']
            step1_metrics = extract_metrics(step1_summary)
            step_data.append(('Step 1: clean_artifacts', step1_metrics, step1_summary))

        # Step 2: eeg_picard (combine ICA arrays - show all)
        ica_arrays = ['icaweights', 'icasphere', 'icawinv']
        for idx, array_name in enumerate(ica_arrays):
            if array_name in summaries:
                array_summary = summaries[array_name]
                array_metrics = extract_metrics(array_summary)
                if idx == 0:
                    step_name = 'Step 2: eeg_picard'
                else:
                    step_name = ''
                step_data.append((f'  {array_name}' if idx > 0 else step_name,
                                array_metrics, array_summary))

        # Step 3: iclabel
        if 'iclabel_classifications' in summaries:
            step3_summary = summaries['iclabel_classifications']
            step3_metrics = extract_metrics(step3_summary)
            step_data.append(('Step 3: iclabel', step3_metrics, step3_summary))

        # Print table
        print(f"\n{'Step':<30} {'Max Abs Diff':<18} {'Mean Abs Diff':<18} {'RMS Diff':<18} {'Max Rel Diff':<18} {'Mismatch %':<15}")
        print("-" * 120)

        for step_name, metrics, _ in step_data:
            print(f"{step_name:<30} {metrics['max_abs_diff']:<18} {metrics['mean_abs_diff']:<18} "
                  f"{metrics['rms_diff']:<18} {metrics['max_rel_diff']:<18} {metrics['mismatch_pct']:<15}")

        print("-" * 120)
        print("\nDetailed summaries:")
        for step_name, _, summary in step_data:
            print(f"\n{step_name}:")
            print(summary)

        print("\n" + "="*80)
        print("Full pipeline test completed successfully!")
        print("="*80 + "\n")


if __name__ == "__main__":
    unittest.main()
