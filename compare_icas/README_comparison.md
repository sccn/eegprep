# ICA Implementation Comparison

This directory contains tools to compare different implementations of the Infomax ICA algorithm.

## Script: compare_ica_versions.py

### Purpose
Runs 6 different ICA implementations on the same EEG dataset and compares their weight matrices.

### Implementations Tested

1. **MNE** (`mne.preprocessing.ICA` with `method='infomax'`)
   - Python implementation from MNE-Python library
   - Uses scipy infomax algorithm

2. **eegprep runica.py**
   - Python port of EEGLAB's runica.m
   - Located in `src/eegprep/runica.py`
   - Achieves 99.97-99.99% parity with MATLAB

3. **EEGLAB runica.m** (via compatibility mode)
   - Original MATLAB implementation
   - Called through EEGPREP's MATLAB engine compatibility layer
   - Requires MATLAB installation

4. **runica_simple.m**
   - Simplified MATLAB reference implementation
   - Located in `runica_c/matlab/runica_simple.m`
   - Used for testing and validation

5. **runica_simple_mex** (C compiled MEX)
   - C-compiled version of runica_simple
   - MEX file: `runica_c/matlab/runica_simple_mex.mexmaca64`
   - Requires MATLAB

6. **binica** (C compiled binary)
   - Standalone C implementation
   - Binary: `binica/ica_darwin` (macOS) or `binica/ica_linux` (Linux)
   - No MATLAB required

### Usage

```bash
# From project root
source .venv/bin/activate
python compare_icas/compare_ica_versions.py
```

### Configuration

Edit the script to change:
- `MAXSTEPS = 100`: Number of ICA iterations
- `DATA_FILE`: Path to .set file (defaults to `runica_c/data/eeglab_data.set`)

### Output

Results are saved to `compare_icas/weights/`:
- `<name>_weights.mat`: Unmixing weight matrix (ncomps × nchans)
- `<name>_sphere.mat`: Sphering/whitening matrix (nchans × nchans)
- `<name>_info.mat`: Contains `lrates` (learning rates) and `n_steps` (iteration count)

### Comparison Metrics

The script compares implementations using:
1. **Correlation coefficient** between flattened weight matrices
2. **Maximum absolute difference** between weight matrices
3. **Iteration count** and final convergence value

### Notes

- All implementations use the same random seed (5489, MATLAB default) for reproducibility
- MATLAB-based implementations (3-5) require MATLAB R2025a to be installed
- binica uses float32 for data storage but float64 for computation
- Component ordering may differ between implementations due to variance-based sorting

### Expected Results

- **eegprep** should closely match MATLAB runica (correlation > 0.99)
- **MNE** may differ more due to different implementation details
- **binica** should match MATLAB when using same parameters
- **MEX versions** should exactly match their MATLAB equivalents

### Troubleshooting

If MATLAB engine fails to start:
- Check MATLAB installation path in `CLAUDE.md`
- Verify MATLAB engine for Python is installed
- Implementations 1, 2, and 6 will still run (no MATLAB required)

If binica fails:
- Check platform-specific binary exists (`ica_darwin` for macOS, `ica_linux` for Linux)
- Verify binary has execute permissions: `chmod +x binica/ica_darwin`
