# pop_epoch

`pop_epoch` extracts epochs from continuous EEG data using selected event types
or event indices.

## Usage

```python
EEG, indices = pop_epoch(EEG, ["stim"], [-0.2, 0.8])
EEG, com = pop_epoch(EEG, ["stim"], [-0.2, 0.8], return_com=True)
```

Calling `pop_epoch(EEG)` opens the EEGPrep dialog. In `eegprep-console`, the
dialog and command-line calls update the shared GUI workspace history.

## Inputs

- `types`: event type string, regular expression string, or a list of event
  types. Use `[]` to epoch around all events.
- `lim`: epoch limits in seconds, `[start, end]`, relative to each selected
  time-locking event.
- `eventindices`: optional event indices to use before event-type filtering.
- `valuelim`: optional amplitude rejection limits. A single value means
  `[-value, value]`.
- `newname`: name for the output dataset.
- `epochinfo`: `yes` or `no`; defaults to `yes`.

## Notes

EEGPrep keeps returned `indices` 0-based for Python callers. The command
history string remains EEGLAB-style so it is readable from the GUI history and
shared console workspace.
