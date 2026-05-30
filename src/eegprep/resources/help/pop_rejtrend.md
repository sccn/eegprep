POP_REJTREND - Reject epochs with linear trends.

Usage:

    EEG = pop_rejtrend(EEG, icacomp, elecrange, winsize, maxslope, min_r)
    EEG, command = pop_rejtrend(EEG, return_com=True)

Set `icacomp=1` for data channels and `icacomp=0` for ICA activations. The
function marks epochs whose selected rows contain a line-like trend exceeding
the slope and fit thresholds.
