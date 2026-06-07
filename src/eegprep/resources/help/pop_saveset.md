# POP_SAVESET - Save an EEG dataset

`pop_saveset` saves an EEGPrep/EEGLAB-style EEG dictionary as a `.set` file.

Usage:

```python
pop_saveset(EEG, "cleaned.set")
pop_saveset(EEG, "cleaned.set", savemode="twofiles")
```

`savemode="twofiles"` writes a `.set` header and an EEGLAB-style `.fdt`
float32 sidecar. The same layout is used by default when
`EEG_OPTIONS["option_savetwofiles"] = 1`.

Use `savemode="resave"` to keep an existing two-file dataset in its `.fdt`
sidecar. A plain save follows the current `option_savetwofiles` setting and
may inline data into the `.set` file when that option is disabled.

The main-window save actions update dataset `filename`, `filepath`, and
`saved` metadata and record the save command in session history. Resaving
multiple selected datasets requires each dataset to already have a filename.

See also: POP_LOADSET
