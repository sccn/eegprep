# POP_EXPORT - Export EEG data to text

`pop_export` writes EEG data or ICA activity to a text file.

Usage:

```python
com = pop_export(EEG, "data.tsv", "transpose", "on")
```

The main-window export action prompts for an output file and records the
resulting command in session history. Use BIDS export for dataset sidecars and
structured folder output.

See also: POP_EXPORTBIDS, POP_EXPEVENTS, POP_WRITEEEG
