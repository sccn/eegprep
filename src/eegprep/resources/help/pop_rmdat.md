# pop_rmdat

Keep or remove continuous data windows around event types.

`pop_rmdat(EEG, events, timelims, invertsel)` uses event types and time limits
in seconds. `invertsel=1` removes selected windows, matching EEGLAB's current
default behavior; `invertsel=0` keeps selected windows.

This function is for continuous data. Use event/epoch selection workflows for
epoched datasets.
