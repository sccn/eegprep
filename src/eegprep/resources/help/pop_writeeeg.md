# POP_WRITEEG - Export EEG data to EDF/BDF/GDF

`pop_writeeeg` writes the current dataset to an external EEG file format.

Usage:

```python
com = pop_writeeeg(EEG, "recording.edf")
com = pop_writeeeg(EEG, "recording.bdf", "TYPE", "BDF")
com = pop_writeeeg(EEG, "recording.gdf", "TYPE", "GDF")
```

The File > Export menu prompts for an output path and records the command in
session history. EDF and BDF are written with their standard 16-bit and 24-bit
sample ranges. GDF uses float64 samples, so signal values are not quantized;
numeric event types and sample timing are stored in the GDF event table. Since
GDF event types are uint16 codes, free-text labels receive file-local codes and
raise a warning that reports the mapping. Use `pop_saveset` for EEGPrep/EEGLAB
`.set` files and `pop_exportbids` for BIDS folder output.

See also: POP_SAVESET, POP_EXPORTBIDS
