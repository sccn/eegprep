POP_REJSPEC - Reject epochs by spectral power.

Usage:

    EEG, rejected = pop_rejspec(EEG, icacomp, "elecrange", [1], "threshold", [-30, 30])
    EEG, command = pop_rejspec(EEG, return_com=True)

Set `icacomp=1` for data channels and `icacomp=0` for ICA activations.
Threshold and frequency limits can be supplied as EEGLAB-style vectors.

Marks are stored in `rejfreq`/`rejfreqE` or `icarejfreq`/`icarejfreqE`.
