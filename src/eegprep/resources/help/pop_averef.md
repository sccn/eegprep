# pop_averef

`pop_averef(EEG)` is a legacy EEGLAB compatibility wrapper for average
reference conversion. EEGPrep delegates the operation to `pop_reref(EEG, [])`
and returns a replayable `pop_averef` history command for users who type the
legacy name.

Use `pop_reref` for new workflows when you need reference-channel, exclusion,
interpolation, or ICA-reference options.
