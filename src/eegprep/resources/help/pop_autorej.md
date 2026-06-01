POP_AUTOREJ - Automatically reject abnormal epochs.

Usage:

    EEG, rejected = pop_autorej(EEG, "nogui", "on")
    EEG, command = pop_autorej(EEG, return_com=True)

`pop_autorej` applies EEGLAB-style large-amplitude, probability, and kurtosis
epoch rejection to epoched data. The GUI path opens EEGPlot/EEGBrowser
inspection by default so users can review and update marks before accepting
rejection.

Important options include `threshold`, `startprob`, `maxrej`, `electrodes`, and
`icacomps`. Rejected epoch numbers are EEGLAB-facing 1-based indices.
