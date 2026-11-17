# pop_eegplot

`pop_eegplot` opens the EEGPrep scrolling browser for the current EEG dataset.
Use `icacomp=1` for channel data and `icacomp=0` for component activations.

Dragging in the browser marks continuous data stretches. A click marks or
clears one whole epoch, and a drag does not extend that mark. Clicking an
existing continuous mark removes it; in electrode-marking mode, clicking a
marked channel toggles only that channel in the `winrej` channel-mask columns.

Calling `pop_eegplot(EEG)` on continuous data, including Tools > Inspect/reject
data by eye, shows the rejection instructions first. Continue opens the
browser. Cancel, or closing the dialog, does not. Passing `reject` explicitly,
as Plot > Channel data (scroll) does, skips the dialog. `show=False` skips it
as well.

With epoched data, `reject=0` stores accepted marks in
`EEG.reject.rejmanual`/`rejmanualE` or the ICA-prefixed counterparts. With
continuous data, `reject=0` stores accepted mark windows in
`EEG.reject.rejmanualwinrej` or `icarejmanualwinrej`, preserving the full
`[start end R G B channel_mask...]` rows for later review. Stored continuous
marks reload automatically the next time `pop_eegplot` opens the browser, so
the normal Inspect/Reject workflow can reject them later. With `reject=1`,
continuous marks are converted through `eegplot2event` and removed with
`eeg_eegrej`; epoched marks are removed with `pop_rejepoch` after confirmation.
NO, or closing that dialog, keeps every epoch. The menu then asks what to do
with the new dataset. Cancel there discards the rejection. The recorded epoch
command uses confirmation flag 0, so replaying it does not ask again.

See `eegplot` help for the core browser options.
