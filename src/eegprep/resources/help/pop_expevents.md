# POP_EXPEVENTS - Export events

`pop_expevents` writes EEG events to a text table.

Usage:

```python
com = pop_expevents(EEG, "events.tsv")
```

The exported table preserves event fields available in the current EEG
dictionary. Use `pop_importevent` to import compatible event tables.

See also: POP_IMPORTEVENT
