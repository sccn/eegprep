POP_JOINTPROB - Reject epochs by joint probability.

Usage:

    EEG, locthresh, globthresh, nrej = pop_jointprob(EEG, icacomp, elecrange, locthresh, globthresh)
    EEG, command = pop_jointprob(EEG, return_com=True)

Set `icacomp=1` for data channels and `icacomp=0` for ICA activations. Local and
global thresholds are standard-deviation cutoffs, matching EEGLAB's rejection
workflow.

Marks are stored in `rejjp`/`rejjpE` or `icarejjp`/`icarejjpE`.
