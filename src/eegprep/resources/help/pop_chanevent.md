# POP_CHANEVENT - Import events from a data channel

`pop_chanevent` detects rising, falling, or both edge types in one or more data
channels and converts them to EEG event records.

Usage:

```python
EEG = pop_chanevent(EEG, 1)
EEG, com = pop_chanevent(EEG, [1, 2], "edge", "leading", return_com=True)
```

Channel indices are EEGLAB-style 1-based values. Continuous 2-D datasets are
supported. Options can delete the event channel after import, replace existing
events, append to existing events, and compute event durations from leading and
trailing edges.

See also: POP_IMPORTEVENT
