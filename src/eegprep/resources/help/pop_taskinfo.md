# POP_TASKINFO - Edit BIDS task metadata

`pop_taskinfo` attaches BIDS task metadata to an EEG or STUDY dictionary.

Usage:

```python
EEG, com = pop_taskinfo(EEG, TaskName="rest")
```

File > BIDS tools opens a dialog for the dataset name, task name, README,
task description, instructions, authors, references, Cognitive Atlas and CogPO
identifiers, institution, cap, reference, ground, montage, amplifier, filters,
and line frequency. Keyword arguments such as `TaskName="rest"` skip the dialog.
Values are stored under `target["etc"]["bids"]["task"]`.

See also: POP_PARTICIPANTINFO, POP_EVENTINFO, POP_EXPORTBIDS
