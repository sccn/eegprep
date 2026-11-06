# POP_WRITEEG - Export EEG data to EDF/BDF/GDF

`pop_writeeeg` writes the current dataset to an external EEG file format.

Usage:

```python
com = pop_writeeeg(EEG, "recording.edf")
com = pop_writeeeg(EEG, "recording.bdf", "TYPE", "BDF")
com = pop_writeeeg(EEG, "recording.gdf", "TYPE", "GDF")
```

File > Export asks for a file name and the EDF, GDF, or BDF format. Epoched
data is concatenated when you accept that prompt. The history command is
`pop_writeeeg(EEG, 'file', 'TYPE', 'EDF');` with no `LASTCOM` prefix. EDF and
BDF keep the original sample count and every event annotation. Samples use the
standard 16-bit and 24-bit ranges. GDF uses float64 samples, so signal values
are not quantized;
numeric event types and sample timing are stored in the GDF event table. Since
GDF event types are uint16 codes, free-text labels receive file-local codes and
raise a warning that reports the mapping. Use `pop_saveset` for EEGPrep/EEGLAB
`.set` files and `pop_exportbids` for BIDS folder output.

See also: POP_SAVESET, POP_EXPORTBIDS
