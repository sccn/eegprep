# pop_eegplot

`pop_eegplot` opens the EEGPrep scrolling browser for the current EEG dataset.
Use `icacomp=1` for channel data and `icacomp=0` for component activations.

Phase 1 is non-mutating: it displays data, overlays events and existing
`winrej` marks, and records a history command, but it does not yet write visual
marks back to `EEG.reject` or remove marked data. See `eegplot` help for the
core browser options.
