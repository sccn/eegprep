POP_REJKURT - Reject epochs by kurtosis.

Usage:

    EEG, locthresh, globthresh, nrej = pop_rejkurt(EEG, icacomp, elecrange, locthresh, globthresh)
    EEG, command = pop_rejkurt(EEG, return_com=True)

Set `icacomp=1` for data channels and `icacomp=0` for ICA activations.
Kurtosis marks are stored in `rejkurt`/`rejkurtE` or
`icarejkurt`/`icarejkurtE`.
