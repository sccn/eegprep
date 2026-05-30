POP_AUTOREJ - Automatically reject abnormal epochs.

Usage:

    EEG, rejected = pop_autorej(EEG, "nogui", "on")
    EEG, command = pop_autorej(EEG, return_com=True)

`pop_autorej` applies EEGLAB-style large-amplitude, probability, and kurtosis
epoch rejection to epoched data. It does not open EEGPlot/EEGBrowser scrolling
inspection windows.

Important options include `threshold`, `startprob`, `maxrej`, `electrodes`, and
`icacomps`. Rejected epoch numbers are EEGLAB-facing 1-based indices.
