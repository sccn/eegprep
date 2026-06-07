# POP_LOADSET - Load an EEGLAB dataset

`pop_loadset` loads EEGLAB/EEGPrep `.set` datasets into an EEG dictionary.

Usage:

```python
EEG = pop_loadset("sample.set")
```

When `EEG_OPTIONS["option_memmapdata"] = 1`, two-file datasets saved with an
`.fdt` sidecar load through a NumPy-compatible memory map. Single-file `.set`
datasets still load into memory.

The main-window "Load existing dataset" action stores the loaded dataset in
the shared `EEGPrepSession`, updates `EEG`, `ALLEEG`, and `CURRENTSET`, and
records the load command in history.

See also: POP_SAVESET, LOADSET
