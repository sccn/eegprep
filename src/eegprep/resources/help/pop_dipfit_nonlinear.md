# pop_dipfit_nonlinear

Fine-fit ICA component dipoles.

EEGPrep refines the selected component with the head model selected in DIPFIT settings. `standardBEM` uses the packaged dipoli leadfield and SciPy BFGS. That BFGS refinement has no inside-brain constraint, so a fit can land outside the brain mesh with a lower residual than EEGLAB's. Sample component 3 does this. `standardBESA` uses the four-shell BESA leadfield and Nelder-Mead with FieldTrip `fminsearch` limits. The fit can update moments only or position plus moment, preserves the EEGLAB-style `EEG["dipfit"]["model"]` fields, and records replayable command history.

The manual dialog mirrors EEGLAB's fields; pressing OK applies the displayed positions/selections and runs the standalone fit.
