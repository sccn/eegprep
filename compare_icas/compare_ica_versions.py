#!/usr/bin/env python3
"""
Compare 6 different runica implementations on the same dataset.

This script runs all available ICA implementations and compares their results:

1. MNE version (Python, mne.preprocessing.ICA with method='infomax')
2. eegprep runica.py version (Python port of EEGLAB runica)
3. EEGLAB runica.m via compatibility mode (MATLAB reference)
4. runica_simple.m via MATLAB engine (MATLAB simplified version)
5. runica_c executable (C compiled version)
6. binica executable (C compiled version)

Configuration (modify at top of script):
- MAXSTEPS: Maximum ICA steps (default: 100)
- EXTENDED: ICA mode - 0=standard (logistic), 1=extended (tanh)
- BINICA_STOP: Convergence threshold for binica (higher = more steps)

Results are saved to compare_icas/weights/ directory:
- <name>_weights.mat: unmixing weight matrix
- <name>_sphere.mat: sphering/whitening matrix
- <name>_info.mat: iteration info (lrates, n_steps)

The script compares weight matrices using correlation and max absolute difference.

Notes:
- runica_c requires recompilation if MAXSTEPS is changed (hardcoded in C source)
- binica stop threshold may need adjustment to reach target step count
"""

import sys
import os
import numpy as np
import scipy.io
import subprocess
import tempfile
from pathlib import Path

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
BINICA_STOP = 1e-9  # Lower stop threshold for binica to run more steps (default: 1e-6)

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


def save_results(name, weights, sphere, lrates, n_steps, extended=False):
    """Save weight and sphere matrices to weights directory."""
    suffix = "_ext" if extended else ""
    weights_file = WEIGHTS_DIR / f"{name}{suffix}_weights.mat"
    sphere_file = WEIGHTS_DIR / f"{name}{suffix}_sphere.mat"
    info_file = WEIGHTS_DIR / f"{name}{suffix}_info.mat"

    scipy.io.savemat(weights_file, {'weights': weights})
    scipy.io.savemat(sphere_file, {'sphere': sphere})
    scipy.io.savemat(info_file, {'lrates': lrates, 'n_steps': n_steps})

    mode_str = "extended" if extended else "standard"
    print(f"  Saved {name} ({mode_str}) results to weights/")
    return weights_file, sphere_file


def run_mne_ica(data):
    """Run MNE ICA implementation."""
    print("\n1. Running MNE ICA...")
    try:
        from mne.preprocessing import ICA

        # MNE uses samples x channels, we have channels x samples
        # MNE ICA uses infomax by default
        # MNE's fit_params allows passing extended parameter to infomax
        fit_params = {'extended': bool(EXTENDED)} if EXTENDED else {}
        ica = ICA(n_components=data.shape[0],
                  method='infomax',
                  max_iter=MAXSTEPS,
                  fit_params=fit_params,
                  random_state=5489)  # Match MATLAB default seed

        # Create fake info for MNE
        import mne
        info = mne.create_info(ch_names=[f'Ch{i+1}' for i in range(data.shape[0])],
                               sfreq=250,  # Arbitrary
                               ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        # Fit ICA
        ica.fit(raw)

        # Extract matrices
        # MNE stores unmixing matrix, we need to extract weights and sphere
        # unmixing = weights @ sphere
        unmixing = ica.unmixing_matrix_

        # MNE doesn't separate weights and sphere in the same way
        # We'll use identity for sphere and put everything in weights
        weights = unmixing
        sphere = np.eye(data.shape[0])

        # MNE doesn't expose lrates or final convergence directly
        # Use a placeholder
        lrates = np.zeros(ica.n_iter_)
        n_steps = ica.n_iter_

        print(f"  Completed ({n_steps} steps)")
        save_results('mne', weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'n_steps': n_steps, 'lrates': lrates}

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def run_eegprep_runica(data):
    """Run eegprep runica.py implementation."""
    print("\n2. Running eegprep runica.py...")
    try:
        # Call runica with maxsteps and extended
        weights, sphere, meanvar, bias, signs, lrates = runica(
            data,
            maxsteps=MAXSTEPS,
            extended=EXTENDED,
            verbose=0
        )

        # Get step count
        n_steps = len(lrates)

        print(f"  Completed ({n_steps} steps)")
        save_results('eegprep', weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'lrates': lrates, 'n_steps': n_steps}

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None




def run_eeglab_runica(data):
    """Run EEGLAB runica.m via compatibility mode."""
    print("\n3. Running EEGLAB runica.m (compatibility mode)...")
    try:
        # Get MATLAB engine
        eeglab = get_eeglab(runtime='MAT', auto_file_roundtrip=False)

        # Save data to temporary file
        temp_data = tempfile.mktemp(suffix='.mat')
        scipy.io.savemat(temp_data, {'data': data})

        # Load and run EEGLAB runica (without pythoncompat to match runica_simple)
        matlab_code = f"""
        load('{temp_data}');
        """
        if EXTENDED:
            matlab_code += f"[weights,sphere,~,~,~,lrates] = runica(data, 'extended', {EXTENDED}, 'maxsteps', {MAXSTEPS}, 'verbose', 'off', 'rndreset', 'off');\n"
        else:
            matlab_code += f"[weights,sphere,~,~,~,lrates] = runica(data, 'maxsteps', {MAXSTEPS}, 'verbose', 'off', 'rndreset', 'off');\n"

        # Set RNG AFTER runica is called (since rndreset=off prevents internal rand('state',0))
        matlab_code = f"rng(5489, 'twister'); % Set before runica\n" + matlab_code

        # Extract results
        temp_out = tempfile.mktemp(suffix='.mat')
        matlab_code += f"save('{temp_out}', 'weights', 'sphere', 'lrates');\n"

        eeglab.eval(matlab_code, nargout=0)

        result = scipy.io.loadmat(temp_out)
        weights = result['weights']
        sphere = result['sphere']
        lrates = result['lrates'].flatten()

        # Cleanup
        os.remove(temp_data)
        os.remove(temp_out)

        n_steps = len(lrates)
        print(f"  Completed ({n_steps} steps)")
        save_results('eeglab_runica', weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'lrates': lrates, 'n_steps': n_steps}

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_runica_simple_matlab(data):
    """Run runica_c/matlab/runica_simple.m."""
    print("\n4. Running runica_simple.m (MATLAB)...")
    try:
        # Get MATLAB engine
        eeglab = get_eeglab(runtime='MAT')

        # Add path
        matlab_dir = str(RUNICA_SIMPLE.parent)
        eeglab.engine.eval(f"addpath('{matlab_dir}');", nargout=0)

        # Save data to temporary file
        temp_data = tempfile.mktemp(suffix='.mat')
        scipy.io.savemat(temp_data, {'data': data})

        # Set random seed and load data
        # runica_simple(data, extended, pca, stop, maxstep, blockint, rndreset)
        eeglab.engine.eval(f"rng(5489, 'twister'); load('{temp_data}');", nargout=0)
        eeglab.engine.eval(
            f"[weights,sphere,meanvar,bias,signs,lrates] = runica_simple(data, {EXTENDED}, 0, 0, {MAXSTEPS}, 0, 0);",
            nargout=0
        )

        # Extract results
        temp_out = tempfile.mktemp(suffix='.mat')
        eeglab.engine.eval(
            f"save('{temp_out}', 'weights', 'sphere', 'lrates');",
            nargout=0
        )

        result = scipy.io.loadmat(temp_out)
        weights = result['weights']
        sphere = result['sphere']
        lrates = result['lrates'].flatten()

        n_steps = len(lrates)

        # Cleanup
        os.remove(temp_data)
        os.remove(temp_out)

        print(f"  Completed ({n_steps} steps)")
        save_results('runica_simple_matlab', weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'lrates': lrates, 'n_steps': n_steps}

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_runica_c(data):
    """Run runica_c executable (C compiled version)."""
    print("\n5. Running runica_c (C executable)...")

    if RUNICA_C_BIN is None or not RUNICA_C_BIN.exists():
        print(f"  SKIPPED: runica_c binary not available for {SYSTEM}")
        return None

    try:
        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        data_file = os.path.join(temp_dir, 'data.fdt')
        wts_file = os.path.join(temp_dir, 'data.wts')
        sph_file = os.path.join(temp_dir, 'data.sph')

        # Write data in float32 format (runica_c expects float32)
        nchans, npoints = data.shape
        data.astype(np.float32).T.tofile(data_file)

        # Run runica_c
        # Arguments: datafile wtsfile sphfile nchans npoints extended
        # NOTE: runica_c has maxsteps hardcoded to 512 in source code
        result = subprocess.run(
            [str(RUNICA_C_BIN), data_file, wts_file, sph_file,
             str(nchans), str(npoints), str(EXTENDED)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"  ERROR: runica_c failed with code {result.returncode}")
            print(f"  stderr: {result.stderr}")
            return None

        # Read weights and sphere
        weights = np.fromfile(wts_file, dtype=np.float64).reshape(nchans, nchans).T
        sphere = np.fromfile(sph_file, dtype=np.float64).reshape(nchans, nchans).T

        # Parse output for step count and wchange values
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
                        # Extract wchange if present
                        if part == 'wchange' and i+1 < len(parts):
                            val_str = parts[i+1].rstrip(',')
                            wchanges.append(float(val_str))
                except:
                    pass

        lrates = np.array(wchanges) if len(wchanges) > 0 else np.zeros(n_steps)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

        print(f"  Completed (~{n_steps} steps)")
        save_results('runica_c', weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'lrates': lrates, 'n_steps': n_steps}

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_binica(data):
    """Run C compiled binica version."""
    print("\n6. Running binica (C compiled)...")

    if BINICA_BIN is None or not BINICA_BIN.exists():
        print(f"  SKIPPED: binica binary not available for {SYSTEM}")
        return None

    try:
        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        data_file = os.path.join(temp_dir, 'data.fdt')
        wts_file = os.path.join(temp_dir, 'data.wts')
        sph_file = os.path.join(temp_dir, 'data.sph')
        config_file = os.path.join(temp_dir, 'ica.sc')

        # Write data in float32 format (binica requirement)
        data.astype(np.float32).T.tofile(data_file)

        nchans, npoints = data.shape

        # Create config file
        with open(config_file, 'w') as f:
            f.write(f"DataFile {data_file}\n")
            f.write(f"chans {nchans}\n")
            f.write(f"datalength {npoints}\n")
            f.write(f"WeightsOutFile {wts_file}\n")
            f.write(f"SphereFile {sph_file}\n")
            f.write(f"seed 1\n")
            f.write(f"doublewrite on\n")
            f.write(f"extended {EXTENDED}\n")
            f.write(f"maxsteps {MAXSTEPS}\n")
            f.write(f"stop {BINICA_STOP:.1e}\n")  # Higher stop threshold to run more steps

        # Run binica
        with open(config_file, 'r') as config:
            result = subprocess.run(
                [str(BINICA_BIN)],
                stdin=config,
                capture_output=True,
                text=True
            )

        if result.returncode != 0:
            print(f"  ERROR: binica failed with code {result.returncode}")
            print(f"  stderr: {result.stderr}")
            return None

        # Debug: print binica output
        if False:  # Set to True for debugging
            print("  binica output (first 50 lines):")
            for i, line in enumerate(result.stdout.split('\n')[:50]):
                print(f"    {line}")

        # Read weights and sphere
        weights = np.fromfile(wts_file, dtype=np.float64).reshape(nchans, nchans).T
        sphere = np.fromfile(sph_file, dtype=np.float64).reshape(nchans, nchans).T

        # Parse output for step count and convergence info
        # binica outputs: "step X - lrate Y, wchange Z, angledelta W deg"
        n_steps = 0
        wchanges = []
        for line in result.stdout.split('\n'):
            # Look for step count
            if line.strip().startswith('step'):
                try:
                    parts = line.split()
                    step_num = int(parts[1])
                    n_steps = max(n_steps, step_num)

                    # Extract wchange if present
                    for i, part in enumerate(parts):
                        if part == 'wchange' and i+1 < len(parts):
                            val_str = parts[i+1].rstrip(',')
                            wchanges.append(float(val_str))
                except:
                    pass

        lrates = np.array(wchanges)
        if n_steps == 0:
            n_steps = len(lrates)  # Fallback to number of wchanges found

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

        print(f"  Completed (~{n_steps} steps)")
        save_results('binica', weights, sphere, lrates, n_steps, extended=bool(EXTENDED))

        return {'weights': weights, 'sphere': sphere, 'lrates': lrates, 'n_steps': n_steps}

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(results):
    """Compare weight matrices from all versions."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    versions = list(results.keys())

    for i, v1 in enumerate(versions):
        if results[v1] is None:
            continue
        for v2 in versions[i+1:]:
            if results[v2] is None:
                continue

            w1 = results[v1]['weights']
            w2 = results[v2]['weights']

            # Compute correlation between weight matrices
            corr = np.corrcoef(w1.flatten(), w2.flatten())[0, 1]

            # Compute max absolute difference
            max_diff = np.max(np.abs(w1 - w2))

            print(f"{v1:20s} vs {v2:20s}: corr={corr:.6f}, max_diff={max_diff:.6e}")

    print("\nIteration counts and final convergence:")
    for name, res in results.items():
        if res and 'n_steps' in res:
            n_steps = res['n_steps']
            if 'lrates' in res and len(res['lrates']) > 0:
                final_val = res['lrates'][-1]
                print(f"  {name:25s}: {n_steps:3d} steps, final convergence = {final_val:.9e}")
            else:
                print(f"  {name:25s}: {n_steps:3d} steps")


def run_all_implementations(data, extended_mode):
    """Run all ICA implementations with given extended mode."""
    global EXTENDED
    EXTENDED = extended_mode

    mode_str = "Extended (tanh)" if extended_mode else "Standard (logistic)"
    print(f"\n{'='*60}")
    print(f"Running {mode_str} ICA")
    print(f"{'='*60}\n")

    results = {}
    results['mne'] = run_mne_ica(data)
    results['eegprep_runica'] = run_eegprep_runica(data)
    results['eeglab_runica'] = run_eeglab_runica(data)
    results['runica_simple_matlab'] = run_runica_simple_matlab(data)
    results['runica_c'] = run_runica_c(data)
    results['binica'] = run_binica(data)

    return results


def main():
    """Main comparison script."""
    print("="*60)
    print("ICA Implementation Comparison")
    print("="*60)
    print(f"Dataset: {DATA_FILE}")
    print(f"Max steps: {MAXSTEPS}")
    print(f"binica stop threshold: {BINICA_STOP:.1e}")
    print(f"Output directory: {WEIGHTS_DIR}")
    print("\nRunning both standard and extended ICA for all implementations...")
    print()

    # Create weights directory
    WEIGHTS_DIR.mkdir(exist_ok=True)

    # Load data
    data, EEG = load_data()

    # Run standard ICA (extended=0)
    results_standard = run_all_implementations(data, extended_mode=0)
    compare_results(results_standard)

    # Run extended ICA (extended=1)
    results_extended = run_all_implementations(data, extended_mode=1)
    compare_results(results_extended)

    print("\n" + "="*60)
    print("All results saved to:", WEIGHTS_DIR)
    print("  Standard ICA: *_weights.mat, *_sphere.mat")
    print("  Extended ICA: *_ext_weights.mat, *_ext_sphere.mat")
    print("="*60)


if __name__ == '__main__':
    main()
