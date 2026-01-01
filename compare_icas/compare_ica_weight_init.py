#!/usr/bin/env python3
"""
Compare ICA implementations with multiple runs using different weight initializations.

This script runs ICA implementations with 10 different random initializations to assess
variability in solutions. Each method can be run separately via command-line argument.

Available methods:
1. mne - MNE version (Python, mne.preprocessing.ICA with method='infomax')
2. eegprep - eegprep runica.py version (Python port of EEGLAB runica)
3. eeglab - EEGLAB runica.m via compatibility mode (MATLAB reference)
4. runica_simple - runica_simple.m via MATLAB engine (MATLAB simplified version)
5. runica_c - runica_c executable (C compiled version)
6. binica - binica executable (C compiled version)

Configuration (modify at top of script):
- MAXSTEPS: Maximum ICA steps (default: 50)
- EXTENDED: ICA mode - 0=standard (logistic), 1=extended (tanh)
- N_RUNS: Number of runs with different initializations (default: 10)

Results are saved to compare_icas/weights/ directory:
- <name>_run<N>_weights.mat: unmixing weight matrix for run N
- <name>_run<N>_sphere.mat: sphering/whitening matrix for run N
- <name>_run<N>_info.mat: iteration info (lrates, n_steps)

Usage:
    python compare_ica_versions.py mne          # Run MNE infomax only
    python compare_ica_versions.py eegprep      # Run eegprep runica only
    python compare_ica_versions.py all          # Run all methods
"""

import sys
import os
import argparse
import numpy as np
import scipy.io
import subprocess
import tempfile
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# Add eegprep to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from eegprep.pop_loadset import pop_loadset
from eegprep.runica import runica
from eegprep.eeglabcompat import get_eeglab

# Configuration
DATA_FILE = Path(__file__).parent / 'runica_c' / 'data' / 'eeglab_data.set'
WEIGHTS_DIR = Path(__file__).parent / 'weights'
MAXSTEPS = 50
EXTENDED = 0  # 0 = standard ICA (logistic), 1 = extended ICA (tanh)
N_RUNS = 10
BASE_SEED = 1000  # Base seed for reproducibility

# Platform detection for binica
import platform
SYSTEM = platform.system()
if SYSTEM == 'Darwin':
    BINICA_BIN = Path(__file__).parent / 'binica' / 'ica_darwin'
elif SYSTEM == 'Linux':
    BINICA_BIN = Path(__file__).parent / 'binica' / 'ica_linux'
else:
    BINICA_BIN = None

# runica_c locations
RUNICA_SIMPLE = Path(__file__).parent / 'runica_c' / 'matlab' / 'runica_simple.m'
RUNICA_C_DIR = Path(__file__).parent / 'runica_c'
if SYSTEM == 'Darwin':
    RUNICA_C_BIN = RUNICA_C_DIR / 'runica_darwin'
elif SYSTEM == 'Linux':
    RUNICA_C_BIN = RUNICA_C_DIR / 'runica_linux'
else:
    RUNICA_C_BIN = None


def load_data():
    """Load EEG dataset and extract data matrix."""
    print(f"Loading data from: {DATA_FILE}")
    EEG = pop_loadset(str(DATA_FILE))
    data = EEG['data']
    print(f"Data shape: {data.shape} (channels x samples)")
    return data, EEG


def save_results(name, run_idx, weights, sphere, lrates, n_steps, extended=False):
    """Save weight and sphere matrices to weights directory."""
    suffix = "_ext" if extended else ""
    weights_file = WEIGHTS_DIR / f"{name}_run{run_idx:02d}{suffix}_weights.mat"
    sphere_file = WEIGHTS_DIR / f"{name}_run{run_idx:02d}{suffix}_sphere.mat"
    info_file = WEIGHTS_DIR / f"{name}_run{run_idx:02d}{suffix}_info.mat"

    scipy.io.savemat(weights_file, {'weights': weights})
    scipy.io.savemat(sphere_file, {'sphere': sphere})
    scipy.io.savemat(info_file, {'lrates': lrates, 'n_steps': n_steps})

    return weights_file, sphere_file


def run_mne_ica(data, run_idx):
    """Run MNE ICA implementation with specific random seed.

    MNE infomax uses:
    - Identity matrix for initial weights
    - Random permutation at each iteration (controlled by random_state)
    """
    print(f"  Run {run_idx+1}/{N_RUNS}...", end=' ', flush=True)
    try:
        from mne.preprocessing import ICA

        # Use different random seed for each run
        seed = BASE_SEED + run_idx

        # MNE uses samples x channels, we have channels x samples
        fit_params = {'extended': bool(EXTENDED)} if EXTENDED else {}
        ica = ICA(n_components=data.shape[0],
                  method='infomax',
                  max_iter=MAXSTEPS,
                  fit_params=fit_params,
                  random_state=seed)

        # Create fake info for MNE
        import mne
        mne.set_log_level('ERROR')
        info = mne.create_info(ch_names=[f'Ch{i+1}' for i in range(data.shape[0])],
                               sfreq=250,
                               ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)

        # Fit ICA
        ica.fit(raw, verbose=False)

        # Extract matrices
        unmixing = ica.unmixing_matrix_
        weights = unmixing
        sphere = np.eye(data.shape[0])

        # MNE doesn't expose lrates
        lrates = np.zeros(ica.n_iter_)
        n_steps = ica.n_iter_

        print(f"{n_steps} steps")
        save_results('mne', run_idx, weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'n_steps': n_steps, 'lrates': lrates}

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_eegprep_runica(data, run_idx):
    """Run eegprep runica.py implementation with specific random seed."""
    print(f"  Run {run_idx+1}/{N_RUNS}...", end=' ', flush=True)
    try:
        seed = BASE_SEED + run_idx
        np.random.seed(seed)

        weights, sphere, meanvar, bias, signs, lrates = runica(
            data,
            maxsteps=MAXSTEPS,
            extended=EXTENDED,
            verbose=0
        )

        n_steps = len(lrates)
        print(f"{n_steps} steps")
        save_results('eegprep', run_idx, weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'lrates': lrates, 'n_steps': n_steps}

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_eeglab_runica(data, run_idx):
    """Run EEGLAB runica.m via compatibility mode with specific random seed."""
    print(f"  Run {run_idx+1}/{N_RUNS}...", end=' ', flush=True)
    try:
        seed = BASE_SEED + run_idx

        eeglab = get_eeglab(runtime='MAT', auto_file_roundtrip=False)

        temp_data = tempfile.mktemp(suffix='.mat')
        scipy.io.savemat(temp_data, {'data': data})

        matlab_code = f"rng({seed}, 'twister');\n"
        matlab_code += f"load('{temp_data}');\n"

        if EXTENDED:
            matlab_code += f"[weights,sphere,~,~,~,lrates] = runica(data, 'extended', {EXTENDED}, 'maxsteps', {MAXSTEPS}, 'verbose', 'off', 'rndreset', 'off');\n"
        else:
            matlab_code += f"[weights,sphere,~,~,~,lrates] = runica(data, 'maxsteps', {MAXSTEPS}, 'verbose', 'off', 'rndreset', 'off');\n"

        temp_out = tempfile.mktemp(suffix='.mat')
        matlab_code += f"save('{temp_out}', 'weights', 'sphere', 'lrates');\n"

        eeglab.eval(matlab_code, nargout=0)

        result = scipy.io.loadmat(temp_out)
        weights = result['weights']
        sphere = result['sphere']
        lrates = result['lrates'].flatten()

        os.remove(temp_data)
        os.remove(temp_out)

        n_steps = len(lrates)
        print(f"{n_steps} steps")
        save_results('eeglab_runica', run_idx, weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'lrates': lrates, 'n_steps': n_steps}

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_runica_simple_matlab(data, run_idx):
    """Run runica_c/matlab/runica_simple.m with specific random seed."""
    print(f"  Run {run_idx+1}/{N_RUNS}...", end=' ', flush=True)
    try:
        seed = BASE_SEED + run_idx

        eeglab = get_eeglab(runtime='MAT')

        matlab_dir = str(RUNICA_SIMPLE.parent)
        eeglab.engine.eval(f"addpath('{matlab_dir}');", nargout=0)

        temp_data = tempfile.mktemp(suffix='.mat')
        scipy.io.savemat(temp_data, {'data': data})

        eeglab.engine.eval(f"rng({seed}, 'twister'); load('{temp_data}');", nargout=0)
        eeglab.engine.eval(
            f"[weights,sphere,meanvar,bias,signs,lrates] = runica_simple(data, {EXTENDED}, 0, 0, {MAXSTEPS}, 0, 0);",
            nargout=0
        )

        temp_out = tempfile.mktemp(suffix='.mat')
        eeglab.engine.eval(f"save('{temp_out}', 'weights', 'sphere', 'lrates');", nargout=0)

        result = scipy.io.loadmat(temp_out)
        weights = result['weights']
        sphere = result['sphere']
        lrates = result['lrates'].flatten()

        n_steps = len(lrates)

        os.remove(temp_data)
        os.remove(temp_out)

        print(f"{n_steps} steps")
        save_results('runica_simple_matlab', run_idx, weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'lrates': lrates, 'n_steps': n_steps}

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_runica_c(data, run_idx):
    """Run runica_c executable with specific random seed.

    Note: runica_c seed is hardcoded in source - limited randomization support.
    """
    print(f"  Run {run_idx+1}/{N_RUNS}...", end=' ', flush=True)

    if RUNICA_C_BIN is None or not RUNICA_C_BIN.exists():
        print(f"SKIPPED: binary not available for {SYSTEM}")
        return None

    try:
        temp_dir = tempfile.mkdtemp()
        data_file = os.path.join(temp_dir, 'data.fdt')
        wts_file = os.path.join(temp_dir, 'data.wts')
        sph_file = os.path.join(temp_dir, 'data.sph')

        nchans, npoints = data.shape
        data.astype(np.float32).T.tofile(data_file)

        result = subprocess.run(
            [str(RUNICA_C_BIN), data_file, wts_file, sph_file,
             str(nchans), str(npoints), str(EXTENDED)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"ERROR: failed with code {result.returncode}")
            return None

        weights = np.fromfile(wts_file, dtype=np.float64).reshape(nchans, nchans).T
        sphere = np.fromfile(sph_file, dtype=np.float64).reshape(nchans, nchans).T

        n_steps = 0
        wchanges = []
        for line in result.stdout.split('\n'):
            if 'step' in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == 'step' and i+1 < len(parts):
                            step_num = int(parts[i+1].rstrip(':,'))
                            n_steps = max(n_steps, step_num)
                        if part == 'wchange' and i+1 < len(parts):
                            val_str = parts[i+1].rstrip(',')
                            wchanges.append(float(val_str))
                except:
                    pass

        lrates = np.array(wchanges) if len(wchanges) > 0 else np.zeros(n_steps)

        import shutil
        shutil.rmtree(temp_dir)

        print(f"{n_steps} steps")
        save_results('runica_c', run_idx, weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'lrates': lrates, 'n_steps': n_steps}

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_binica(data, run_idx):
    """Run binica executable with specific random seed."""
    print(f"  Run {run_idx+1}/{N_RUNS}...", end=' ', flush=True)

    if BINICA_BIN is None or not BINICA_BIN.exists():
        print(f"SKIPPED: binary not available for {SYSTEM}")
        return None

    try:
        seed = BASE_SEED + run_idx

        temp_dir = tempfile.mkdtemp()
        data_file = os.path.join(temp_dir, 'data.fdt')
        wts_file = os.path.join(temp_dir, 'data.wts')
        sph_file = os.path.join(temp_dir, 'data.sph')
        config_file = os.path.join(temp_dir, 'ica.sc')

        data.astype(np.float32).T.tofile(data_file)

        nchans, npoints = data.shape

        with open(config_file, 'w') as f:
            f.write(f"DataFile {data_file}\n")
            f.write(f"chans {nchans}\n")
            f.write(f"datalength {npoints}\n")
            f.write(f"WeightsOutFile {wts_file}\n")
            f.write(f"SphereFile {sph_file}\n")
            f.write(f"seed {seed}\n")
            f.write(f"doublewrite on\n")
            f.write(f"extended {EXTENDED}\n")
            f.write(f"maxsteps {MAXSTEPS}\n")
            f.write(f"stop 1.0e-9\n")

        with open(config_file, 'r') as config:
            result = subprocess.run(
                [str(BINICA_BIN)],
                stdin=config,
                capture_output=True,
                text=True
            )

        if result.returncode != 0:
            print(f"ERROR: failed with code {result.returncode}")
            return None

        weights = np.fromfile(wts_file, dtype=np.float64).reshape(nchans, nchans).T
        sphere = np.fromfile(sph_file, dtype=np.float64).reshape(nchans, nchans).T

        n_steps = 0
        wchanges = []
        for line in result.stdout.split('\n'):
            if line.strip().startswith('step'):
                try:
                    parts = line.split()
                    step_num = int(parts[1])
                    n_steps = max(n_steps, step_num)

                    for i, part in enumerate(parts):
                        if part == 'wchange' and i+1 < len(parts):
                            val_str = parts[i+1].rstrip(',')
                            wchanges.append(float(val_str))
                except:
                    pass

        lrates = np.array(wchanges)
        if n_steps == 0:
            n_steps = len(lrates)

        import shutil
        shutil.rmtree(temp_dir)

        print(f"{n_steps} steps")
        save_results('binica', run_idx, weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'lrates': lrates, 'n_steps': n_steps}

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_amari_distance(W1, S1, W2, S2):
    """
    Compute AMARI distance between two ICA solutions.

    Lower is better. 0 = identical up to permutation and scaling.
    """
    U1 = W1 @ S1
    U2 = W2 @ S2

    P = U1 @ np.linalg.inv(U2)
    n = P.shape[0]

    row_sum = 0.0
    for i in range(n):
        row_max = np.max(np.abs(P[i, :]))
        row_sum += np.sum(np.abs(P[i, :])) / row_max - 1.0

    col_sum = 0.0
    for j in range(n):
        col_max = np.max(np.abs(P[:, j]))
        col_sum += np.sum(np.abs(P[:, j])) / col_max - 1.0

    amari = (row_sum + col_sum) / (2.0 * n)
    return amari


def compute_component_correlation(W1, S1, W2, S2):
    """
    Compute component correlations after optimal matching.

    Returns average, min, and max correlation across all matched components.
    Uses Hungarian algorithm to find optimal component pairing.
    """
    U1 = W1 @ S1
    U2 = W2 @ S2

    # Get mixing matrices
    A1 = np.linalg.pinv(U1)
    A2 = np.linalg.pinv(U2)

    n = A1.shape[1]

    # Compute correlation matrix between all component pairs
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr = np.corrcoef(A1[:, i], A2[:, j])[0, 1]
            corr_matrix[i, j] = np.abs(corr)

    # Find optimal matching using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-corr_matrix)
    matched_corrs = corr_matrix[row_ind, col_ind]

    return np.mean(matched_corrs), np.min(matched_corrs), np.max(matched_corrs)


def run_method_multiple_times(method_name, data):
    """Run a specific method N_RUNS times with different initializations."""

    method_map = {
        'mne': run_mne_ica,
        'eegprep': run_eegprep_runica,
        'eeglab': run_eeglab_runica,
        'runica_simple': run_runica_simple_matlab,
        'runica_c': run_runica_c,
        'binica': run_binica,
    }

    if method_name not in method_map:
        print(f"ERROR: Unknown method '{method_name}'")
        print(f"Available methods: {', '.join(method_map.keys())}")
        return []

    print(f"\nRunning {method_name} ({N_RUNS} runs with different initializations):")
    print(f"  Max steps: {MAXSTEPS}")
    print(f"  Extended: {bool(EXTENDED)}")
    print(f"  Base seed: {BASE_SEED}")
    print()

    method_func = method_map[method_name]
    results = []

    for run_idx in range(N_RUNS):
        result = method_func(data, run_idx)
        results.append(result)

    return results


def analyze_results(method_name, results):
    """Analyze variability across multiple runs."""
    print(f"\n{'='*80}")
    print(f"Analysis for {method_name}")
    print(f"{'='*80}")

    valid_results = [r for r in results if r is not None]
    if len(valid_results) == 0:
        print("No valid results to analyze")
        return

    n_valid = len(valid_results)
    print(f"\nNumber of valid runs: {n_valid}/{len(results)}")

    # Step count statistics
    step_counts = [r['n_steps'] for r in valid_results]
    print(f"\nStep counts:")
    print(f"  Mean: {np.mean(step_counts):.1f}")
    print(f"  Std:  {np.std(step_counts):.1f}")
    print(f"  Min:  {np.min(step_counts)}")
    print(f"  Max:  {np.max(step_counts)}")

    # AMARI Distance Analysis
    print(f"\n{'='*80}")
    print(f"AMARI Distance (0 = identical up to permutation, lower is better)")
    print(f"{'='*80}")

    amari_distances = []
    for i in range(n_valid):
        for j in range(i+1, n_valid):
            W1 = valid_results[i]['weights']
            S1 = valid_results[i]['sphere']
            W2 = valid_results[j]['weights']
            S2 = valid_results[j]['sphere']

            amari = compute_amari_distance(W1, S1, W2, S2)
            amari_distances.append(amari)

    print(f"\nPairwise AMARI distances:")
    print(f"  Mean: {np.mean(amari_distances):.6f}")
    print(f"  Std:  {np.std(amari_distances):.6f}")
    print(f"  Min:  {np.min(amari_distances):.6f}")
    print(f"  Max:  {np.max(amari_distances):.6f}")

    # Component Correlation After Optimal Matching
    print(f"\n{'='*80}")
    print(f"Component Correlation After Optimal Matching (higher is better)")
    print(f"{'='*80}")

    mean_corrs = []
    min_corrs = []
    max_corrs = []

    for i in range(n_valid):
        for j in range(i+1, n_valid):
            W1 = valid_results[i]['weights']
            S1 = valid_results[i]['sphere']
            W2 = valid_results[j]['weights']
            S2 = valid_results[j]['sphere']

            mean_c, min_c, max_c = compute_component_correlation(W1, S1, W2, S2)
            mean_corrs.append(mean_c)
            min_corrs.append(min_c)
            max_corrs.append(max_c)

    print(f"\nAverage component correlations (after optimal matching):")
    print(f"  Mean of means: {np.mean(mean_corrs):.6f}")
    print(f"  Std of means:  {np.std(mean_corrs):.6f}")
    print(f"  Min of means:  {np.min(mean_corrs):.6f}")
    print(f"  Max of means:  {np.max(mean_corrs):.6f}")

    print(f"\nWorst matched component per pair:")
    print(f"  Mean of mins: {np.mean(min_corrs):.6f}")
    print(f"  Min of mins:  {np.min(min_corrs):.6f}")

    # Detailed pairwise comparison (sorted by AMARI distance)
    print(f"\n{'='*80}")
    print(f"Detailed Pairwise Comparisons (sorted by AMARI distance)")
    print(f"{'='*80}")

    pairwise_data = []
    for i in range(n_valid):
        for j in range(i+1, n_valid):
            W1 = valid_results[i]['weights']
            S1 = valid_results[i]['sphere']
            W2 = valid_results[j]['weights']
            S2 = valid_results[j]['sphere']

            amari = compute_amari_distance(W1, S1, W2, S2)
            mean_c, min_c, max_c = compute_component_correlation(W1, S1, W2, S2)

            pairwise_data.append({
                'run1': i,
                'run2': j,
                'amari': amari,
                'mean_corr': mean_c,
                'min_corr': min_c,
                'max_corr': max_c
            })

    # Sort by AMARI distance (most similar first)
    pairwise_data.sort(key=lambda x: x['amari'])

    print(f"\n{'Pair':12s} {'AMARI':>12s} {'Mean Corr':>12s} {'Min Corr':>12s} {'Max Corr':>12s}")
    print("-" * 60)
    for data in pairwise_data[:10]:  # Show top 10 most similar
        pair_str = f"Run {data['run1']:02d}-{data['run2']:02d}"
        print(f"{pair_str:12s} {data['amari']:12.6f} {data['mean_corr']:12.6f} "
              f"{data['min_corr']:12.6f} {data['max_corr']:12.6f}")

    if len(pairwise_data) > 10:
        print(f"... ({len(pairwise_data) - 10} more pairs)")
        print(f"\nMost dissimilar pairs:")
        print(f"\n{'Pair':12s} {'AMARI':>12s} {'Mean Corr':>12s} {'Min Corr':>12s} {'Max Corr':>12s}")
        print("-" * 60)
        for data in pairwise_data[-5:]:  # Show top 5 most dissimilar
            pair_str = f"Run {data['run1']:02d}-{data['run2']:02d}"
            print(f"{pair_str:12s} {data['amari']:12.6f} {data['mean_corr']:12.6f} "
                  f"{data['min_corr']:12.6f} {data['max_corr']:12.6f}")


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(
        description='Compare ICA implementations with multiple random initializations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available methods:
  mne            - MNE infomax (Python)
  eegprep        - eegprep runica (Python)
  eeglab         - EEGLAB runica.m (MATLAB)
  runica_simple  - runica_simple.m (MATLAB)
  runica_c       - runica_c (C executable)
  binica         - binica (C executable)
  all            - Run all methods

Examples:
  python compare_ica_versions.py mne
  python compare_ica_versions.py all
        """
    )
    parser.add_argument('method',
                        help='Method to run (mne, eegprep, eeglab, runica_simple, runica_c, binica, or all)')

    args = parser.parse_args()

    print("="*60)
    print("ICA Implementation Comparison - Multiple Initializations")
    print("="*60)
    print(f"Dataset: {DATA_FILE}")
    print(f"Max steps: {MAXSTEPS}")
    print(f"Number of runs per method: {N_RUNS}")
    print(f"Output directory: {WEIGHTS_DIR}")
    print()

    # Create weights directory
    WEIGHTS_DIR.mkdir(exist_ok=True)

    # Load data
    data, EEG = load_data()

    # Run requested method(s)
    if args.method == 'all':
        methods = ['mne', 'eegprep', 'eeglab', 'runica_simple', 'runica_c', 'binica']
    else:
        methods = [args.method]

    for method in methods:
        results = run_method_multiple_times(method, data)
        analyze_results(method, results)

    print("\n" + "="*60)
    print(f"Results saved to: {WEIGHTS_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
