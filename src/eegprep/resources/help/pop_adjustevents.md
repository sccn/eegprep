# POP_ADJUSTEVENTS - Adjust event latencies

`pop_adjustevents` shifts event latencies by a specified number of milliseconds
or samples.

Usage:

```python
EEG = pop_adjustevents(EEG, events=["stim"], milliseconds=20)
EEG, com = pop_adjustevents(EEG, samples=-2, return_com=True)
```

Calling `pop_adjustevents(EEG)` opens the interactive dialog. Leave the event
type field empty to adjust all events, or select one or more event types. The
dialog keeps milliseconds and sample offsets synchronized from the dataset
sampling rate.

Boundary events are protected by default. Use the force option only when the
requested shift is known to be valid for the dataset.

Event latencies remain EEGLAB-style 1-based sample positions in the EEG
dictionary.
