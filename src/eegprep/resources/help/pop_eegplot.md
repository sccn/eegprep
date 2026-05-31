# pop_eegplot

`pop_eegplot` opens the EEGPrep scrolling browser for the current EEG dataset.
Use `icacomp=1` for channel data and `icacomp=0` for component activations.

Dragging in the browser marks continuous data stretches or whole epochs.
Clicking an existing mark removes it; in electrode-marking mode, clicking a
marked channel toggles only that channel in the `winrej` channel-mask columns.

With epoched data, `reject=0` stores accepted marks in
`EEG.reject.rejmanual`/`rejmanualE` or the ICA-prefixed counterparts. With
continuous data, `reject=0` stores accepted mark windows in
`EEG.reject.rejmanualwinrej` or `icarejmanualwinrej`, preserving the full
`[start end R G B channel_mask...]` rows for later review. Stored continuous
marks reload automatically the next time `pop_eegplot` opens the browser, so
the normal Inspect/Reject workflow can reject them later. With `reject=1`,
continuous marks are converted through `eegplot2event` and removed with
`eeg_eegrej`; epoched marks are removed with `pop_rejepoch`.

See `eegplot` help for the core browser options.
