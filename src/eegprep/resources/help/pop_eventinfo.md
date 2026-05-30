# POP_EVENTINFO - Edit BIDS event metadata

`pop_eventinfo` attaches BIDS event metadata to an EEG or STUDY dictionary.

The File > BIDS tools menu prompts for `key=value` lines and stores the values
under `target["etc"]["bids"]["event"]`. The function returns the updated
target and a history command.

Usage:

```python
EEG, com = pop_eventinfo(EEG, trial_type="stim")
```

This action edits metadata only; it does not import or export event tables.

See also: POP_TASKINFO, POP_PARTICIPANTINFO, POP_EXPORTBIDS
