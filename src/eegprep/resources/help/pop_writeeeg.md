# POP_WRITEEG - Export EEG data to EDF/BDF/GDF

`pop_writeeeg` writes the current dataset to an external EEG file format.

Usage:

```python
com = pop_writeeeg(EEG, "recording.edf")
```

The File > Export menu prompts for an EDF, BDF, or GDF output path and records
the command in session history. Use `pop_saveset` for EEGPrep/EEGLAB `.set`
files and `pop_exportbids` for BIDS folder output.

See also: POP_SAVESET, POP_EXPORTBIDS
