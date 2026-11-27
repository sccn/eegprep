POP_SELECTCOMPS - Mark ICA components for rejection.

Usage:

    EEG = pop_selectcomps(EEG, compnum, reject=[...])
    EEG, command = pop_selectcomps(EEG, return_com=True)

The dialog shows one button per component, 35 per page. Clicking a button opens
`pop_prop` and an ACCEPT/REJECT control. OK stores the flags in
`EEG.reject.gcompreject`. Cancel leaves the flags unchanged. Set thresholds
and See comp. stats stay disabled until `pop_icathresh` and
`EEG.stats.compenta` exist.

Use `pop_subcomp` to remove marked components from the data.
