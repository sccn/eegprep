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

Channel index note:

Numeric channel numbers are 1-based in the GUI, Python API, and command history. Type `1` for the first channel. For example, `exclude=[2, 6]` (or `2 6` in the dialog) excludes the second and sixth channels. Convert NumPy row indices by adding 1 when passing them to `pop_reref`. Empty reference metadata is displayed as `unknown`.

Inputs:

- `EEG`: input EEG dataset.
- `ref`: `[]` or `None` for average reference, a 1-based channel number, a sequence of 1-based channel numbers, a channel label, or a sequence of channel labels.

Optional inputs:

- `interpchan`: channel locations or 1-based numbers into `EEG["urchanlocs"]` to interpolate before re-referencing, `[]` to infer removed channels, or `"off"`.
- `exclude`: 1-based channel numbers excluded from re-referencing.
- `keepref`: `"on"` keeps reference channels in the data; `"off"` removes explicit reference channels.
- `refloc`: previous reference-channel location to reconstruct, or its 1-based number in the non-fiducial `EEG["chaninfo"]["nodatchans"]` list.
- `refica`: `"on"`, `"off"`, `"backwardcomp"`, or `"remove"` for ICA handling.
- `huber`: Huber threshold in microvolts.
- `method`: rereferencing algorithm. EEGPrep currently supports EEGLAB's `"standard"` method.

Output:

- `EEGOUT`: re-referenced EEG dataset.

See also: REREF, POP_INTERP
