# MATLAB Test Fixtures

This directory contains MATLAB/Octave parity scripts, helper scripts, and
MATLAB-only fixtures used by the Python `unittest` suite.

- `computeg.m` and `spheric_spline.m` are reference helpers for
  `tests/test_eeg_interp.py`.
- `test_matlab_func_args*.m` are MATLAB engine argument-marshalling fixtures
  used by exploratory notebooks and compatibility checks.
- `*_compare.m` and `*_compare_helper.py` scripts are manual parity/debug
  tools for comparing EEGPrep outputs against EEGLAB.

Run these scripts from this directory unless the script says otherwise.
Relative paths assume the repository root is two levels up.
