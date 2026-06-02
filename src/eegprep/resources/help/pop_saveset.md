# POP_SAVESET - Save an EEG dataset

`pop_saveset` saves an EEGPrep/EEGLAB-style EEG dictionary as a `.set` file.

Usage:

```python
pop_saveset(EEG, "cleaned.set")
```

The main-window save actions update dataset `filename`, `filepath`, and
`saved` metadata and record the save command in session history. Resaving
multiple selected datasets requires each dataset to already have a filename.

See also: POP_LOADSET
