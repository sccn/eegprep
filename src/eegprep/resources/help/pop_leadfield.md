# pop_leadfield

Distributed source-model leadfield workflows are not implemented. This entry point raises `NotImplementedError` and does not change the dataset.

For equivalent-dipole localization, configure `pop_dipfit_settings`, then use `pop_dipfit_gridsearch`, `pop_dipfit_nonlinear`, or `pop_multifit`. These functions support the packaged MNI BEM and four-shell BESA models without MATLAB.
