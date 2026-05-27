POP_EDITSET - Edit EEG dataset structure fields.

Usage:

    EEG = pop_editset(EEG)
    EEG = pop_editset(EEG, 'setname', name, 'subject', subject, ...)

Calling `pop_editset(EEG)` opens the dataset information dialog. Command-line
calls update dataset metadata directly.

Common fields:

- `setname`: dataset name.
- `subject`: subject code.
- `condition`: task condition.
- `group`: subject group.
- `run`: run number.
- `session`: session number.
- `srate`: sampling rate in Hz.
- `pnts`: time points per epoch.
- `xmin`: epoch start time in seconds.
- `comments`: dataset comments.

Notes:

- Changing `EEG.ref` only edits metadata; use `pop_reref` to re-reference data.
- Channel-location and ICA file-picking workflows are handled by the channel
  metadata phase. Programmatic direct assignments for `chanlocs`, `icaweights`,
  `icasphere`, and `icachansind` are supported.

See also: POP_COMMENTS, POP_REREF, POP_CHANEDIT
