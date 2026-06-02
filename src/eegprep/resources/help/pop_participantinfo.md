# POP_PARTICIPANTINFO - Edit BIDS participant metadata

`pop_participantinfo` attaches BIDS participant metadata to an EEG or STUDY
dictionary.

Usage:

```python
EEG, com = pop_participantinfo(EEG, participant_id="sub-01")
```

The File > BIDS tools menu prompts for `key=value` lines and stores the values
under `target["etc"]["bids"]["participant"]`.

See also: POP_TASKINFO, POP_EVENTINFO, POP_EXPORTBIDS
