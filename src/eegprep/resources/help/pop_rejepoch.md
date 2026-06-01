POP_REJEPOCH - Remove marked epochs.

Usage:

    EEG = pop_rejepoch(EEG, tmprej, confirm)
    EEG, command = pop_rejepoch(EEG, tmprej, return_com=True)

`tmprej` may be a boolean vector over epochs or a list of EEGLAB-facing 1-based
epoch numbers. If omitted, `EEG.reject.rejglobal` is used.

Review marks with EEGPrep rejection dialogs or the scrolling EEGPlot/EEGBrowser
workflows before removing epochs.
