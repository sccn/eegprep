# POP_IMPORTEVENT - Import event information

`pop_importevent` imports event records from a text table or record sequence.

Usage:

```python
EEG = pop_importevent(EEG, "event", "events.tsv")
EEG, com = pop_importevent(EEG, "event", records, return_com=True)
```

Imported events are normalized into EEGLAB-style event dictionaries with
1-based sample latencies. Use the append option to preserve existing events and
extend `urevent`; otherwise EEGPrep replaces the event table.

See also: POP_CHANEVENT, POP_IMPORTPRES, POP_IMPORTERPLAB
