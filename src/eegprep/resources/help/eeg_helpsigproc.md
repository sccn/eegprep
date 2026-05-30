# EEG_HELPSIGPROC - Signal processing functions

Signal-processing functions implement lower-level numerical operations used by
EEGPrep's `pop_*` wrappers.

Implemented surfaces include filtering, resampling helpers, ICA wrappers,
topographic interpolation, epoch rejection helpers, spectral summaries, and
plotting backends. These functions are ordinary Python APIs and should be
called with explicit arguments in scripts.

Use `pop_*` wrappers when you want EEGLAB-style dialogs, history commands, or
main-window session synchronization.

See also: POP_EEGFILT, POP_EEGFILTNEW, POP_RESAMPLE, POP_RUNICA, TOPOPLOT
