# POP_EXPEVENTS - Export events

`pop_expevents` writes EEG events to a text table.

Usage:

```python
com = pop_expevents(EEG, "events.tsv")
```

The table starts with a `number` column, then the event fields in the order
they appear on the events. Numeric values use six digits after the decimal
point. `urevent` is written 1-based. The default unit is `samples`. Pass
`unit="seconds"` to write `latency` and `duration` in seconds. File > Export
asks for the file name only, the same as the EEGLAB dialog.

See also: POP_IMPORTEVENT
