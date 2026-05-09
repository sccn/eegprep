POP_REREF - Convert an EEG dataset to average reference or to a new common reference channel.

Usage:

    EEGOUT = pop_reref(EEG)
    EEGOUT = pop_reref(EEG, ref, 'key', 'value', ...)

Calling `pop_reref(EEG)` opens the interactive re-reference dialog.

Graphic interface:

- "Compute average reference" converts the data to average reference.
- "Huber average ref. with threshold" uses a Huber average reference threshold in microvolts.
- "Re-reference data to channel(s)" enables entry or selection of reference channel labels or indices.
- "Interpolate removed channel(s)" interpolates removed channels before computing the re-reference.
- "Retain ref. channel(s) in data" keeps explicit reference channels in the output.
- "Exclude channel indices (EMG, EOG)" excludes channels from the reference calculation.
- "Add old ref. channel back to the data" reconstructs an old reference channel when its location is known.

Inputs:

- `EEG`: input EEG dataset.
- `ref`: `[]` or `None` for average reference, a channel index, a sequence of channel indices, a channel label, or a sequence of channel labels.

Optional inputs:

- `interpchan`: channels to interpolate before re-referencing, `[]` to infer removed channels, or `"off"`.
- `exclude`: channel indices excluded from re-referencing.
- `keepref`: `"on"` keeps reference channels in the data; `"off"` removes explicit reference channels.
- `refloc`: previous reference-channel location to reconstruct.
- `refica`: `"on"`, `"off"`, `"backwardcomp"`, or `"remove"` for ICA handling.
- `huber`: Huber threshold in microvolts.

Output:

- `EEGOUT`: re-referenced EEG dataset.

See also: REREF, POP_INTERP
