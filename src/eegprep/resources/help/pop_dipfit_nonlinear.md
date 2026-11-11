# pop_dipfit_nonlinear

Fine-fit ICA component dipoles.

EEGPrep refines the selected component with the head model selected in DIPFIT settings. `standardBEM` uses the packaged dipoli leadfield and SciPy BFGS. `standardBESA` uses the single-sphere leadfield and Nelder-Mead. The fit can update moments only or position plus moment, preserves the EEGLAB-style `EEG["dipfit"]["model"]` fields, and records replayable command history.

The manual dialog mirrors EEGLAB's fields; pressing OK applies the displayed positions/selections and runs the standalone fit.
