POP_REJSPEC - Reject epochs by spectral power.

Usage:

    EEG, rejected = pop_rejspec(EEG, icacomp, "elecrange", [1], "threshold", [-30, 30])
    EEG, command = pop_rejspec(EEG, return_com=True)

Set `icacomp=1` for data channels and `icacomp=0` for ICA activations.
Threshold and frequency limits can be supplied as EEGLAB-style vectors.

Marks are stored in `rejfreq`/`rejfreqE` or `icarejfreq`/`icarejfreqE`.

Multitaper power is converted to decibels with `10*log10`. EEGLAB
`pop_rejspec.m:310` uses `10*log` (natural log), so the same numeric
threshold marks fewer epochs here. On `eeglab_data_epochs_ica.set`,
15–30 Hz at ±35 dB marks no data epochs; EEGLAB marks epoch 37. At ±30 dB,
EEGPrep marks no data epochs and no component epochs; EEGLAB marks data
epoch 37 and components 13, 14, 15, 26, and 79. FFT spectra already use
`10*log10` in both programs.
