# POP_PARTICIPANTINFO - Edit BIDS participant metadata

`pop_participantinfo` attaches BIDS participant metadata to an EEG or STUDY
dictionary.

Usage:

```python
EEG, com = pop_participantinfo(EEG, participant_id="sub-01")
```

File > BIDS tools opens a dialog whose first field is the subject code, then
Gender, Age, Group, head circumference, and artefact description. The subject
code is required. Keyword arguments skip the dialog. Values are stored under
`target["etc"]["bids"]["participant"]`.

See also: POP_TASKINFO, POP_EVENTINFO, POP_EXPORTBIDS
