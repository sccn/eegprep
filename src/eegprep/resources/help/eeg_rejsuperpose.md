EEG_REJSUPERPOSE - Combine stored rejection marks.

Usage:

    EEG = eeg_rejsuperpose(EEG, typerej, Rmanual, Rthres, Rconst, Rent, Rkurt, Rfreq, Rothertype)

`typerej` follows EEGLAB: `1` combines data-channel rejection fields and `0`
combines ICA-component rejection fields. The combined result is written to
`EEG.reject.rejglobal` and `EEG.reject.rejglobalE`.

EEGPrep uses this helper from rejection menus and browser-backed review flows
to keep data-channel and ICA-component rejection marks synchronized with
EEGLAB-compatible fields.
