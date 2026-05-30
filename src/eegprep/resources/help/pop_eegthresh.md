POP_EEGTHRESH - Reject epochs by extreme values.

Usage:

    EEG, rejected = pop_eegthresh(EEG, icacomp, elecrange, negthresh, posthresh, starttime, endtime)
    EEG, command = pop_eegthresh(EEG, return_com=True)

Set `icacomp=1` for data channels and `icacomp=0` for ICA activations. Channel,
component, and epoch numbers shown to users are 1-based.

The function stores marks in EEGLAB-compatible `EEG.reject.*` fields and can
optionally remove marked epochs.
