POP_SELECTCOMPS - Mark ICA components for rejection.

Usage:

    EEG = pop_selectcomps(EEG, compnum, reject=[...])
    EEG, command = pop_selectcomps(EEG, return_com=True)

The dialog shows one button per component, 35 per page. Clicking a button opens
`pop_prop` with Cancel, ACCEPT/REJECT, and OK. OK in `pop_prop` returns the
mark to the grid; Cancel there drops it. OK in the grid stores the flags in
`EEG.reject.gcompreject`. Cancel in the grid leaves the flags unchanged. Set thresholds
and See comp. stats stay disabled until `pop_icathresh` and
`EEG.stats.compenta` exist.

Use `pop_subcomp` to remove marked components from the data.
