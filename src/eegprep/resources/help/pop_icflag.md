POP_ICFLAG - Flag ICLabel components for rejection.

Usage:

    EEG = pop_icflag(EEG, thresholds)
    EEG, command = pop_icflag(EEG, thresholds, return_com=True)

Inputs:

- `EEG`: EEGPrep/EEGLAB-style dataset with ICLabel classifications.
- `thresholds`: 7-by-2 matrix ordered as Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, and Other.

Graphical interface:

Calling `pop_icflag(EEG)` opens an EEGLAB-style threshold dialog. Blank
fields ignore a class. By default, Muscle and Eye components with probability
between 0.9 and 1 are flagged.

Behavior:

- Run `pop_iclabel` before `pop_icflag`.
- Flagged components are stored in `EEG.reject.gcompreject`.
- Existing rejection fields are preserved.
- This function flags components only; use `pop_subcomp` to remove them from the data.

Example:

    thresholds = [[None, None], [0.9, 1], [0.9, 1], [None, None], [None, None], [None, None], [None, None]]
    EEG = pop_icflag(EEG, thresholds)
