"""
Utility for stage-by-stage pipeline comparison between Python and MATLAB.
"""
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import tempfile


def _match_components_by_scalp_maps(icawinv_py: np.ndarray, icawinv_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Match Python components to MATLAB by maximizing scalp map correlations.

    Returns:
        reorder_idx: Indices to reorder Python components to match MATLAB
        correlations: Correlation for each matched pair
    """
    n_comps = min(icawinv_py.shape[1], icawinv_mat.shape[1])

    # Compute correlation matrix between all pairs
    corr_matrix = np.zeros((n_comps, n_comps))
    for i in range(n_comps):
        for j in range(n_comps):
            corr_matrix[i, j] = abs(np.corrcoef(icawinv_py[:, i], icawinv_mat[:, j])[0, 1])

    # Greedy matching: for each MATLAB component, find best Python match
    reorder_idx = np.zeros(n_comps, dtype=int)
    correlations = np.zeros(n_comps)
    used = set()

    for mat_idx in range(n_comps):
        # Find best unused Python component
        best_py_idx = -1
        best_corr = -1
        for py_idx in range(n_comps):
            if py_idx not in used and corr_matrix[py_idx, mat_idx] > best_corr:
                best_corr = corr_matrix[py_idx, mat_idx]
                best_py_idx = py_idx
        reorder_idx[mat_idx] = best_py_idx
        correlations[mat_idx] = best_corr
        used.add(best_py_idx)

    return reorder_idx, correlations


def compare_ica_decompositions(py_file: str, mat_file: str, subject: str) -> Dict[str, any]:
    """Compare ICA decompositions between Python and MATLAB.

    Returns dict with: avg_corr, min_corr, max_corr, scalp_map_file
    """
    from eegprep import pop_loadset, eeg_checkset_strict_mode

    with eeg_checkset_strict_mode(False):
        EEG_py = pop_loadset(py_file)
        EEG_mat = pop_loadset(mat_file)

    if EEG_py.get('icawinv') is None or EEG_mat.get('icawinv') is None:
        return None

    icawinv_py = EEG_py['icawinv']
    icawinv_mat = EEG_mat['icawinv']

    # Match components by scalp map correlation
    reorder_idx, correlations = _match_components_by_scalp_maps(icawinv_py, icawinv_mat)

    # Save scalp map comparison using Python topoplot
    scalp_map_file = py_file.replace('_py.set', '_scalp_maps_python.png')
    try:
        import matplotlib.pyplot as plt
        from eegprep import topoplot

        n_comps_to_plot = min(10, len(correlations))
        fig, axes = plt.subplots(2, n_comps_to_plot, figsize=(3*n_comps_to_plot, 6))
        if n_comps_to_plot == 1:
            axes = axes.reshape(2, 1)

        # Get channel locations
        chanlocs_mat = EEG_mat['chanlocs']
        chanlocs_py = EEG_py['chanlocs']

        for i in range(n_comps_to_plot):
            # MATLAB scalp map
            ax = axes[0, i] if n_comps_to_plot > 1 else axes[0]
            plt.sca(ax)  # Set current axis
            topoplot(icawinv_mat[:, i], chanlocs_mat, noplot='off', ELECTRODES='off')
            ax.set_title(f'MAT IC{i+1}', fontsize=10)
            ax.axis('off')

            # Python scalp map (reordered to match MATLAB)
            ax = axes[1, i] if n_comps_to_plot > 1 else axes[1]
            plt.sca(ax)  # Set current axis
            topoplot(icawinv_py[:, reorder_idx[i]], chanlocs_py, noplot='off', ELECTRODES='off')
            ax.set_title(f'Py IC{i+1}\n(r={correlations[i]:.3f})', fontsize=10)
            ax.axis('off')

        plt.suptitle(f'ICA Component Scalp Maps: MATLAB (top) vs Python (bottom) - Python Topoplot', fontsize=12, y=0.98)
        plt.tight_layout()
        plt.savefig(scalp_map_file, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create Python topoplot scalp map: {e}")
        scalp_map_file = None

    # Save scalp map comparison using MATLAB topoplot
    scalp_map_file_matlab = os.path.abspath(py_file.replace('_py.set', '_scalp_maps_matlab.png'))
    try:
        from eegprep.eeglabcompat import get_eeglab
        import tempfile

        n_comps_to_plot = min(10, len(correlations))

        # Create temporary files for MATLAB to process (use absolute paths)
        temp_py = tempfile.NamedTemporaryFile(suffix='_py.set', delete=False)
        temp_mat = tempfile.NamedTemporaryFile(suffix='_mat.set', delete=False)
        temp_py.close()
        temp_mat.close()

        # Save reordered Python components to temp file
        EEG_py_reordered = EEG_py.copy()
        EEG_py_reordered['icawinv'] = icawinv_py[:, reorder_idx]
        from eegprep import pop_saveset
        pop_saveset(EEG_py_reordered, temp_py.name)
        pop_saveset(EEG_mat, temp_mat.name)

        # Convert correlations to numpy array for MATLAB
        corr_array = np.array(correlations[:n_comps_to_plot], dtype=np.float64)

        # Use MATLAB to create topoplot (use absolute paths)
        eeglab = get_eeglab('MATLAB')
        eeglab.generate_ica_comparison_plot(
            os.path.abspath(temp_mat.name),
            os.path.abspath(temp_py.name),
            scalp_map_file_matlab,
            n_comps_to_plot,
            corr_array
        )

        # Clean up temp files
        os.unlink(temp_py.name)
        os.unlink(temp_mat.name)

    except Exception as e:
        print(f"Warning: Could not create MATLAB topoplot scalp map: {e}")
        import traceback
        traceback.print_exc()
        scalp_map_file_matlab = None

    return {
        'avg_corr': float(np.mean(correlations)),
        'min_corr': float(np.min(correlations)),
        'max_corr': float(np.max(correlations)),
        'n_components': len(correlations),
        'reorder_idx': reorder_idx,
        'scalp_map_file': scalp_map_file,
        'scalp_map_file_matlab': scalp_map_file_matlab
    }


def compare_iclabel_classifications(py_file: str, mat_file: str, ica_reorder: Optional[np.ndarray] = None) -> Dict[str, any]:
    """Compare ICLabel classifications between Python and MATLAB.

    Args:
        ica_reorder: Component reordering from ICA comparison

    Returns dict with: avg_prob_diff, max_prob_diff, n_flagged_py, n_flagged_mat
    """
    from eegprep import pop_loadset, eeg_checkset_strict_mode

    with eeg_checkset_strict_mode(False):
        EEG_py = pop_loadset(py_file)
        EEG_mat = pop_loadset(mat_file)

    if 'etc' not in EEG_py or 'ic_classification' not in EEG_py['etc']:
        return None
    if 'etc' not in EEG_mat or 'ic_classification' not in EEG_mat['etc']:
        return None

    ic_py = EEG_py['etc']['ic_classification']['ICLabel']['classifications']
    ic_mat = EEG_mat['etc']['ic_classification']['ICLabel']['classifications']

    # Reorder Python classifications to match MATLAB
    if ica_reorder is not None:
        ic_py = ic_py[ica_reorder, :]

    # Compute probability differences
    prob_diff = np.abs(ic_py - ic_mat)

    # Count flagged components using pop_icflag criteria: [NaN NaN;0.9 1;0.9 1;NaN NaN;NaN NaN;NaN NaN;NaN NaN]
    # Class order: Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other
    # Flag if Muscle > 0.9 OR Eye > 0.9
    flagged_py = np.sum((ic_py[:, 1] > 0.9) | (ic_py[:, 2] > 0.9))
    flagged_mat = np.sum((ic_mat[:, 1] > 0.9) | (ic_mat[:, 2] > 0.9))

    return {
        'avg_prob_diff': float(np.mean(prob_diff)),
        'max_prob_diff': float(np.max(prob_diff)),
        'n_flagged_py': int(flagged_py),
        'n_flagged_mat': int(flagged_mat),
        'n_components': ic_py.shape[0]
    }


def compute_data_after_component_removal(iclabel_file: str) -> np.ndarray:
    """Compute EEG data after removing flagged ICA components.

    Returns the data with bad components removed (but before interpolation).
    """
    from eegprep import pop_loadset, eeg_checkset_strict_mode

    with eeg_checkset_strict_mode(False):
        EEG = pop_loadset(iclabel_file)

    # Get flagged components
    if 'etc' not in EEG or 'ic_classification' not in EEG['etc']:
        return EEG['data']

    ic_class = EEG['etc']['ic_classification']['ICLabel']['classifications']
    # Flag components using pop_icflag criteria: Muscle > 0.9 OR Eye > 0.9
    # Class order: Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other
    flagged = (ic_class[:, 1] > 0.9) | (ic_class[:, 2] > 0.9)
    keep_comps = ~flagged  # Keep non-flagged components

    # Remove flagged components from data
    if EEG.get('icaweights') is not None and EEG.get('icasphere') is not None:
        # Reconstruct data without bad components
        W = EEG['icaweights'] @ EEG['icasphere']
        A = np.linalg.pinv(W)

        # Zero out bad components
        A_clean = A.copy()
        A_clean[:, ~keep_comps] = 0

        # Reconstruct
        data_clean = A_clean @ W @ EEG['data'].reshape(EEG['nbchan'], -1)
        return data_clean.reshape(EEG['data'].shape)

    return EEG['data']


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
        # Exclude ICA (9) and ICLabel (10) - they don't change data, only metadata
        stages = [1, 2, 3, 8, 11, 12, 13, 14]

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

    # Main results table
    lines = []
    lines.append("=" * 105)
    lines.append("Stage-by-Stage Python vs MATLAB Comparison (Cumulative Differences)")
    lines.append("=" * 105)

    for basename in sorted(basenames):
        results = []

        # Collect standard stage comparisons
        for stage in stages:
            py_file = [f for f in all_files if f.startswith(f'{basename}_stage{stage:02d}_') and f.endswith('_py.set')]
            mat_file = [f for f in all_files if f.startswith(f'{basename}_stage{stage:02d}_') and f.endswith('_mat.set')]

            if not py_file or not mat_file:
                continue

            py_path = os.path.join(stage_dir, py_file[0])
            mat_path = os.path.join(stage_dir, mat_file[0])

            try:
                metrics = compare_eeg_data(py_path, mat_path)
                results.append((stage, stage_names.get(stage, f'Stage{stage}'), metrics))
            except Exception as e:
                print(f"Error comparing stage {stage} for {basename}: {e}")

        if not results:
            continue

        # Display main table for this subject
        lines.append(f"\n{basename}")
        lines.append("-" * 105)
        lines.append(f"{'Stage':<6} {'Name':<15} {'Cumul Max':<15} {'Cumul Mean':<15} {'Cumul RMS':<15} {'Incr Max':<15} {'Incr Mean':<15}")
        lines.append("-" * 105)

        prev_max = 0.0
        prev_mean = 0.0

        for stage_num, stage_name, metrics in results:
            incr_max = metrics['max_abs'] - prev_max
            incr_mean = metrics['mean_abs'] - prev_mean

            lines.append(f"{stage_num:<6} {stage_name:<15} {metrics['max_abs']:<15.3e} "
                        f"{metrics['mean_abs']:<15.3e} {metrics['rms']:<15.3e} "
                        f"{incr_max:<15.3e} {incr_mean:<15.3e}")

            prev_max = metrics['max_abs']
            prev_mean = metrics['mean_abs']

            # Add computed row for component removal (after stage 8, before stage 11)
            if stage_num == 8:
                py_icl = [f for f in all_files if f.startswith(f'{basename}_stage10_') and f.endswith('_py.set')]
                mat_icl = [f for f in all_files if f.startswith(f'{basename}_stage10_') and f.endswith('_mat.set')]

                if py_icl and mat_icl:
                    try:
                        # Compute data after component removal
                        data_py_clean = compute_data_after_component_removal(os.path.join(stage_dir, py_icl[0]))
                        data_mat_clean = compute_data_after_component_removal(os.path.join(stage_dir, mat_icl[0]))

                        # Compute difference metrics
                        diff = data_py_clean - data_mat_clean
                        ic_metrics = {
                            'max_abs': float(np.max(np.abs(diff))),
                            'mean_abs': float(np.mean(np.abs(diff))),
                            'rms': float(np.sqrt(np.mean(diff**2)))
                        }

                        incr_max_ic = ic_metrics['max_abs'] - prev_max
                        incr_mean_ic = ic_metrics['mean_abs'] - prev_mean

                        lines.append(f"{'9+10':<6} {'AfterICRemoval':<15} {ic_metrics['max_abs']:<15.3e} "
                                    f"{ic_metrics['mean_abs']:<15.3e} {ic_metrics['rms']:<15.3e} "
                                    f"{incr_max_ic:<15.3e} {incr_mean_ic:<15.3e}")

                        prev_max = ic_metrics['max_abs']
                        prev_mean = ic_metrics['mean_abs']
                    except Exception as e:
                        lines.append(f"{'9+10':<6} {'AfterICRemoval':<15} {'Error':<15} {'Error':<15} {'Error':<15} {'N/A':<15} {'N/A':<15}")

        # ICA Analysis
        py_ica = [f for f in all_files if f.startswith(f'{basename}_stage09_') and f.endswith('_py.set')]
        mat_ica = [f for f in all_files if f.startswith(f'{basename}_stage09_') and f.endswith('_mat.set')]

        if py_ica and mat_ica:
            lines.append("")
            lines.append("  ICA Decomposition Comparison:")
            try:
                ica_result = compare_ica_decompositions(
                    os.path.join(stage_dir, py_ica[0]),
                    os.path.join(stage_dir, mat_ica[0]),
                    basename
                )
                if ica_result:
                    lines.append(f"    Components: {ica_result['n_components']}, "
                                f"Avg corr: {ica_result['avg_corr']:.3f}, "
                                f"Min corr: {ica_result['min_corr']:.3f}, "
                                f"Max corr: {ica_result['max_corr']:.3f}")
                    if ica_result['scalp_map_file']:
                        lines.append(f"    Scalp maps (Python topoplot): {os.path.basename(ica_result['scalp_map_file'])}")
                    if ica_result.get('scalp_map_file_matlab'):
                        lines.append(f"    Scalp maps (MATLAB topoplot): {os.path.basename(ica_result['scalp_map_file_matlab'])}")
                else:
                    lines.append("    No ICA data available")
            except Exception as e:
                lines.append(f"    Error: {e}")
                ica_result = None

            # ICLabel Analysis
            py_icl = [f for f in all_files if f.startswith(f'{basename}_stage10_') and f.endswith('_py.set')]
            mat_icl = [f for f in all_files if f.startswith(f'{basename}_stage10_') and f.endswith('_mat.set')]

            if py_icl and mat_icl and ica_result:
                lines.append("")
                lines.append("  ICLabel Classification Comparison:")
                try:
                    icl_result = compare_iclabel_classifications(
                        os.path.join(stage_dir, py_icl[0]),
                        os.path.join(stage_dir, mat_icl[0]),
                        ica_result.get('reorder_idx')
                    )
                    if icl_result:
                        lines.append(f"    Avg prob diff: {icl_result['avg_prob_diff']:.3f}, "
                                    f"Max prob diff: {icl_result['max_prob_diff']:.3f}")
                        lines.append(f"    Flagged for removal: Python={icl_result['n_flagged_py']}, "
                                    f"MATLAB={icl_result['n_flagged_mat']}")
                    else:
                        lines.append("    No ICLabel data available")
                except Exception as e:
                    lines.append(f"    Error: {e}")

    lines.append("=" * 105)
    return '\n'.join(lines)


def save_comparison_report(stage_dir: str, comparison_table: str, study: str, subjects: list, runs: list) -> None:
    """Save comparison results as a markdown report.

    Args:
        stage_dir: Directory containing stage files
        comparison_table: The formatted comparison table text
        study: Study name
        subjects: List of subjects analyzed
        runs: List of runs analyzed
    """
    from datetime import datetime

    report_path = os.path.join(stage_dir, 'comparison_report.md')

    # Collect PNG files
    png_files = sorted([f for f in os.listdir(stage_dir) if f.endswith('.png')])

    with open(report_path, 'w') as f:
        f.write(f"# Stage-by-Stage Pipeline Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Study:** {study}\n\n")
        f.write(f"**Subjects:** {', '.join(map(str, subjects))}\n\n")
        f.write(f"**Runs:** {', '.join(map(str, runs))}\n\n")
        f.write(f"**Analysis Directory:** `{stage_dir}`\n\n")

        f.write("## Comparison Results\n\n")
        f.write("```\n")
        f.write(comparison_table)
        f.write("\n```\n\n")

        if png_files:
            f.write("## ICA Scalp Map Comparisons\n\n")
            for png in png_files:
                subject = png.split('_stage')[0]
                f.write(f"### {subject}\n\n")
                f.write(f"![{subject}]({png})\n\n")

        f.write("## Files in This Analysis\n\n")
        f.write("### Staged EEG Files\n\n")
        set_files = sorted([f for f in os.listdir(stage_dir) if f.endswith('.set')])
        for set_file in set_files:
            f.write(f"- `{set_file}`\n")

        if png_files:
            f.write("\n### Scalp Map Visualizations\n\n")
            for png in png_files:
                f.write(f"- `{png}`\n")

    print(f"Saved comparison report to: {report_path}")
