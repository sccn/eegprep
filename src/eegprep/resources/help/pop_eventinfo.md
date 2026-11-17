# POP_EVENTINFO - Edit BIDS event metadata

`pop_eventinfo` attaches BIDS event metadata to an EEG or STUDY dictionary.

File > BIDS tools opens a dialog for the `onset`, `duration`, `sample`,
`value`, `trial_type`, `stim_file`, and `response_time` descriptions. Keyword
arguments skip the dialog. Values are stored under `target["etc"]["bids"]["event"]`.
The function returns the updated target and a history command.

Usage:

```python
EEG, com = pop_eventinfo(EEG, trial_type="stim")
```

This action edits metadata only; it does not import or export event tables.

See also: POP_TASKINFO, POP_PARTICIPANTINFO, POP_EXPORTBIDS
