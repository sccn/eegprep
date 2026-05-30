# POP_IMPORTEPOCH - Import epoch metadata

`pop_importepoch` imports one row of metadata per epoch from a text table.

Usage:

```python
EEG = pop_importepoch(EEG, "epochs.tsv", ["condition", "rt"])
EEG, com = pop_importepoch(EEG, "epochs.tsv", return_com=True)
```

The current dataset must be epoched, and the number of imported rows must match
`EEG["trials"]`. Options can control header rows, latency fields, duration
fields, type fields, and whether existing events are cleared.

See also: POP_IMPORTEVENT, POP_EPOCH
