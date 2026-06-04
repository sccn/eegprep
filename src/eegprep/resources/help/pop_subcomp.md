POP_SUBCOMP - Remove ICA components from EEG data.

Usage:

    EEG = pop_subcomp(EEG, components)
    EEG = pop_subcomp(EEG, [])
    EEG, command = pop_subcomp(EEG, components, return_com=True)

Inputs:

- `EEG`: EEGPrep/EEGLAB-style dataset with an ICA decomposition.
- `components`: 1-based component numbers to remove. Use `[]` or omit the argument to remove components flagged in `EEG.reject.gcompreject`.
- `keepcomp`: set to `1` to keep the supplied components and remove all others.

Graphical interface:

Calling `pop_subcomp(EEG)` opens a compact EEGLAB-style dialog. The first
field lists components to remove. The second field lists components to retain
and overrides the removal field when filled.

Behavior:

- ICA weights and sphere matrices must already be present.
- Data are projected back using the remaining components.
- ICA activations and rejection flags are cleared after removal, matching EEGLAB's component-removal workflow.
- With `plotag=1`, EEGPrep opens a before/after scrolling browser: black traces
  show the original channel data and red `data2` traces show the data after
  removing the selected components.
