# pop_multifit

Automatically fit DIPFIT models for multiple ICA components.

EEGPrep combines native spherical grid search, nonlinear refinement, residual-variance rejection, optional outside-head removal for spherical coordinates, and optional `pop_dipplot` plotting. The output is stored in `EEG["dipfit"]["model"]` using EEGLAB-compatible fields.

Bilateral fits use the same x-axis symmetry convention for MNI coordinates and y-axis symmetry for spherical coordinates that EEGLAB uses before calling FieldTrip.
