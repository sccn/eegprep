# pop_dipfit_nonlinear

Fine-fit ICA component dipoles.

EEGPrep refines the selected component using a standalone spherical leadfield and SciPy optimization. It can fit moments only or refine position plus moment, preserves the EEGLAB-style `EEG["dipfit"]["model"]` fields, and records replayable command history.

The manual dialog mirrors EEGLAB's fields; pressing OK applies the displayed positions/selections and runs the standalone fit.
