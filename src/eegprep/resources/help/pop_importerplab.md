# POP_IMPORTERPLAB - Import ERPLAB event-list text

`pop_importerplab` imports ERPLAB-style event-list text through EEGPrep's
generic event importer.

Usage:

```python
EEG = pop_importerplab(EEG, "events.txt")
EEG, com = pop_importerplab(EEG, "events.txt", return_com=True)
```

The default field order is latency then type. Override `fields` or `timeunit`
when your event list uses a different table layout.

See also: POP_IMPORTEVENT
