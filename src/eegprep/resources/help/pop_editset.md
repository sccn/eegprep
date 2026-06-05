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
- `data` may be assigned directly as a Python array or loaded from an existing
  file path. Use `dataformat` to force the import format, or omit it to infer
  from the extension.
- `chanlocs` may be assigned directly as channel-location structures or loaded
  from an existing channel-location file path supported by `readlocs`.
- `icaweights`, `icasphere`, and `icachansind` may be assigned directly or
  loaded from numeric file paths.
- MATLAB workspace expressions such as `rawdata`, `locs`, or `icaweights`
  are not evaluated by EEGPrep. Pass Python values directly or provide concrete
  file paths so the command history is replayable in `eegprep-console`.

See also: POP_COMMENTS, POP_REREF, POP_CHANEDIT
