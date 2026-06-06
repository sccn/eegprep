# POP_LIMORESULTS - LIMO result limitation

EEGPrep does not browse or compute EEGLAB LIMO result files in standalone
Python.

Calling `pop_limoresults` raises a clear `NotImplementedError`. This avoids
silently treating external-toolbox results as native EEGPrep outputs.

See also: POP_LIMO, EEG_HELPSTUDY
