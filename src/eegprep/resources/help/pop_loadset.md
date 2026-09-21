# POP_LOADSET - Load an EEGLAB dataset

`pop_loadset` loads EEGLAB/EEGPrep `.set` datasets into an EEG dictionary.

Usage:

```python
EEG = pop_loadset("sample.set")
metadata = pop_loadset("sample.set", loadmode="info")
channel_10 = pop_loadset("sample.set", loadmode=10)
```

`loadmode="info"` loads the dataset metadata while leaving `EEG["data"]` as
the stored data filename (or `"in set file"`). An integer or sequence loads
only those channels using EEGLAB's 1-based channel numbers and clears the ICA
fields, which no longer describe the selected channel matrix.

When `EEG_OPTIONS["option_memmapdata"] = 1`, two-file datasets saved with an
`.fdt` sidecar load through a NumPy-compatible memory map. Single-file `.set`
datasets still load into memory.

The main-window "Load existing dataset" action stores the loaded dataset in
the shared `EEGPrepSession`, updates `EEG`, `ALLEEG`, and `CURRENTSET`, and
records the load command in history.

See also: POP_SAVESET, LOADSET
