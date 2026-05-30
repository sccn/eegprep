# POP_TASKINFO - Edit BIDS task metadata

`pop_taskinfo` attaches BIDS task metadata to an EEG or STUDY dictionary.

Usage:

```python
EEG, com = pop_taskinfo(EEG, TaskName="rest")
```

The File > BIDS tools menu prompts for `key=value` lines and stores the values
under `target["etc"]["bids"]["task"]`.

See also: POP_PARTICIPANTINFO, POP_EVENTINFO, POP_EXPORTBIDS
