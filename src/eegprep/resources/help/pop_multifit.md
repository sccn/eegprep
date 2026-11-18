# pop_multifit

Automatically fit DIPFIT models for multiple ICA components.

EEGPrep combines grid search, nonlinear refinement, residual-variance rejection, optional outside-head removal for spherical coordinates, and optional `pop_dipplot` plotting. The output is stored in `EEG["dipfit"]["model"]` using EEGLAB-compatible fields. `standardBEM` uses the packaged dipoli leadfield and EEGLAB's default 24 by 24 by 12 grid. Its BFGS refinement has no inside-brain constraint, so a fit can land outside the brain mesh with a lower residual than EEGLAB's. Sample component 3 does this. `standardBESA` uses the four-shell BESA leadfield.

The plot-options field accepts MATLAB pairs such as `'normlen' 'on'`. When plotting is checked, the dipole figure is shown.

Bilateral fits use the same x-axis symmetry convention for MNI coordinates and y-axis symmetry for spherical coordinates that EEGLAB uses before calling FieldTrip.
