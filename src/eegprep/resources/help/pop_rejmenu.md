POP_REJMENU - Combine and remove stored rejection marks.

Usage:

    EEG = pop_rejmenu(EEG)
    EEG, command = pop_rejmenu(EEG, return_com=True)

The EEGPrep dialog combines non-browser rejection marks into
`EEG.reject.rejglobal` using `eeg_rejsuperpose`. It can then remove the marked
epochs with `pop_rejepoch`.

Use the scrolling EEGPlot/EEGBrowser buttons to review data or component marks
interactively before updating marks or rejecting epochs.
