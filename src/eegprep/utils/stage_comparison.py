"""
Utility for stage-by-stage pipeline comparison between Python and MATLAB.
"""
import os
import numpy as np
from typing import Dict, List, Tuple
import tempfile


def compare_eeg_data(py_file: str, mat_file: str) -> Dict[str, float]:
    """Compare two EEG .set files and return difference metrics.

    Args:
        py_file: Path to Python-generated .set file
        mat_file: Path to MATLAB-generated .set file

    Returns:
        Dictionary with difference metrics
    """
    from eegprep import pop_loadset, eeg_checkset_strict_mode

    with eeg_checkset_strict_mode(False):
        EEG_py = pop_loadset(py_file)
        EEG_mat = pop_loadset(mat_file)

    # Compute difference metrics on data
    diff = EEG_py['data'] - EEG_mat['data']

    return {
        'max_abs': float(np.max(np.abs(diff))),
        'mean_abs': float(np.mean(np.abs(diff))),
        'rms': float(np.sqrt(np.mean(diff**2))),
        'shape_py': EEG_py['data'].shape,
        'shape_mat': EEG_mat['data'].shape,
    }


def run_staged_pipeline_python(root: str, stage_dir: str, **kwargs) -> List[str]:
    """Run bids_preproc and save intermediate stages.

    Args:
        root: BIDS root or file path
        stage_dir: Directory to save intermediate stages
        **kwargs: Arguments passed to bids_preproc

    Returns:
        List of saved file paths for each stage
    """
    from eegprep import bids_preproc, pop_saveset, pop_loadset
    from eegprep.utils.bids import gen_derived_fpath

    # Stage mapping: (output_file, stage_params)
    stages = [
        ('stage01_import_py.set', {'OnlyChannelsWithPosition': False, 'OnlyModalities': [],
                                    'SamplingRate': None, 'WithInterp': False, 'WithPicard': False,
                                    'WithICLabel': False, 'EpochEvents': None, 'CommonAverageReference': False}),
        ('stage02_chansel_py.set', {'OnlyModalities': ['EEG'], 'SamplingRate': None, 'WithInterp': False,
                                     'WithPicard': False, 'WithICLabel': False, 'EpochEvents': None, 'CommonAverageReference': False}),
        ('stage03_resample_py.set', {'OnlyModalities': ['EEG'], 'SamplingRate': 128, 'WithInterp': False,
                                      'WithPicard': False, 'WithICLabel': False, 'EpochEvents': None, 'CommonAverageReference': False}),
        ('stage08_window_py.set', {'OnlyModalities': ['EEG'], 'SamplingRate': 128, 'WithInterp': False,
                                    'WithPicard': False, 'WithICLabel': False, 'EpochEvents': None, 'CommonAverageReference': False}),
        ('stage09_ica_py.set', {'OnlyModalities': ['EEG'], 'SamplingRate': 128, 'WithInterp': False,
                                 'WithPicard': True, 'WithICLabel': False, 'EpochEvents': None, 'CommonAverageReference': False}),
        ('stage10_iclabel_py.set', {'OnlyModalities': ['EEG'], 'SamplingRate': 128, 'WithInterp': False,
                                     'WithPicard': True, 'WithICLabel': True, 'EpochEvents': None, 'CommonAverageReference': False}),
        ('stage11_interp_py.set', {'OnlyModalities': ['EEG'], 'SamplingRate': 128, 'WithInterp': True,
                                    'WithPicard': True, 'WithICLabel': True, 'EpochEvents': None, 'CommonAverageReference': False}),
        ('stage12_epoch_py.set', {'OnlyModalities': ['EEG'], 'SamplingRate': 128, 'WithInterp': True,
                                   'WithPicard': True, 'WithICLabel': True, 'EpochEvents': [], 'EpochLimits': [-0.2, 0.5], 'CommonAverageReference': False}),
        ('stage13_baseline_py.set', {'OnlyModalities': ['EEG'], 'SamplingRate': 128, 'WithInterp': True,
                                      'WithPicard': True, 'WithICLabel': True, 'EpochEvents': [], 'EpochLimits': [-0.2, 0.5],
                                      'EpochBaseline': [-0.2, 0], 'CommonAverageReference': False}),
        ('stage14_reref_py.set', {'OnlyModalities': ['EEG'], 'SamplingRate': 128, 'WithInterp': True,
                                   'WithPicard': True, 'WithICLabel': True, 'EpochEvents': [], 'EpochLimits': [-0.2, 0.5],
                                   'EpochBaseline': [-0.2, 0], 'CommonAverageReference': True}),
    ]

    saved_files = []
    base_kwargs = {k: v for k, v in kwargs.items() if k not in ['OnlyChannelsWithPosition', 'OnlyModalities',
                   'SamplingRate', 'WithInterp', 'WithPicard', 'WithICLabel', 'EpochEvents', 'EpochLimits',
                   'EpochBaseline', 'CommonAverageReference']}

    for stage_file, stage_params in stages:
        stage_path = os.path.join(stage_dir, stage_file)
        if os.path.exists(stage_path):
            print(f"Stage file exists: {stage_path}")
            saved_files.append(stage_path)
            continue

        # Merge parameters
        run_kwargs = {**base_kwargs, **stage_params}
        run_kwargs['ReturnData'] = True
        run_kwargs['SkipIfPresent'] = True

        # Run pipeline
        print(f"Running Python pipeline to generate {stage_file}...")
        result = bids_preproc(root, **run_kwargs)

        if result and len(result) > 0:
            EEG = result[0] if isinstance(result, list) else result
            if EEG is not None:
                pop_saveset(EEG, stage_path)
                saved_files.append(stage_path)
                print(f"Saved {stage_path}")

    return saved_files


def generate_comparison_table(stage_dir: str, stages: List[int] = None) -> str:
    """Generate comparison table for all stage files.

    Args:
        stage_dir: Directory containing stage files
        stages: List of stage numbers to compare (default: all available)

    Returns:
        Formatted table string
    """
    if stages is None:
        stages = [1, 2, 3, 8, 9, 10, 11, 12, 13, 14]

    stage_names = {
        1: 'Import', 2: 'ChanSel', 3: 'Resample', 8: 'CleanArtifacts',
        9: 'ICA', 10: 'ICLabel', 11: 'Interp', 12: 'Epoch', 13: 'Baseline', 14: 'Reref'
    }

    # First, collect all unique basenames
    all_files = os.listdir(stage_dir)
    basenames = set()
    for f in all_files:
        if '_stage' in f and f.endswith('.set'):
            basenames.add(f.split('_stage')[0])

    # Organize results by basename
    results_by_file = {}
    for basename in sorted(basenames):
        results_by_file[basename] = []

        for stage in stages:
            # Find matching Python and MATLAB files for this basename and stage
            py_file = [f for f in all_files if f.startswith(f'{basename}_stage{stage:02d}_') and f.endswith('_py.set')]
            mat_file = [f for f in all_files if f.startswith(f'{basename}_stage{stage:02d}_') and f.endswith('_mat.set')]

            if not py_file or not mat_file:
                continue

            py_path = os.path.join(stage_dir, py_file[0])
            mat_path = os.path.join(stage_dir, mat_file[0])

            try:
                metrics = compare_eeg_data(py_path, mat_path)
                results_by_file[basename].append((stage, stage_names.get(stage, f'Stage{stage}'), metrics))
            except Exception as e:
                print(f"Error comparing stage {stage} for {basename}: {e}")

    # Format table - show each file separately
    lines = []
    lines.append("=" * 105)
    lines.append("Stage-by-Stage Python vs MATLAB Comparison (Cumulative Differences)")
    lines.append("=" * 105)

    for basename in sorted(results_by_file.keys()):
        if not results_by_file[basename]:
            continue

        lines.append(f"\nFile: {basename}")
        lines.append("-" * 105)
        lines.append(f"{'Stage':<6} {'Name':<15} {'Cumul Max':<15} {'Cumul Mean':<15} {'Cumul RMS':<15} {'Incr Max':<15} {'Incr Mean':<15}")
        lines.append("-" * 105)

        prev_max = 0.0
        prev_mean = 0.0

        for stage_num, stage_name, metrics in results_by_file[basename]:
            incr_max = metrics['max_abs'] - prev_max
            incr_mean = metrics['mean_abs'] - prev_mean

            lines.append(f"{stage_num:<6} {stage_name:<15} {metrics['max_abs']:<15.3e} "
                        f"{metrics['mean_abs']:<15.3e} {metrics['rms']:<15.3e} "
                        f"{incr_max:<15.3e} {incr_mean:<15.3e}")

            prev_max = metrics['max_abs']
            prev_mean = metrics['mean_abs']

    lines.append("=" * 105)

    return '\n'.join(lines)
