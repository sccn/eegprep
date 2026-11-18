# pop_dipfit_nonlinear

Fine-fit ICA component dipoles.

EEGPrep refines the selected component with the four-shell BESA leadfield when the head model is `standardBESA`. MNI and BEM settings still use an infinite-medium leadfield. Position refinement uses Nelder-Mead with the same iteration and tolerance limits as FieldTrip `fminsearch`. It can fit moments only or refine position plus moment, preserves the `EEG["dipfit"]["model"]` fields, and records replayable command history.

The manual dialog mirrors EEGLAB's fields; pressing OK applies the displayed positions/selections and runs the standalone fit.
