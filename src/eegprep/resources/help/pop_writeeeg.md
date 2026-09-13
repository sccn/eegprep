# POP_WRITEEG - Export EEG data to EDF/BDF

`pop_writeeeg` writes the current dataset to an external EEG file format.

Usage:

```python
com = pop_writeeeg(EEG, "recording.edf")
com = pop_writeeeg(EEG, "recording.bdf", "TYPE", "BDF")
```

The File > Export menu prompts for an output path and records the command in
session history. EDF and BDF are written with their standard 16-bit and 24-bit
sample ranges. GDF writing is not yet available and raises a clear error. Use
`pop_saveset` for EEGPrep/EEGLAB `.set` files and `pop_exportbids` for BIDS
folder output.

See also: POP_SAVESET, POP_EXPORTBIDS
