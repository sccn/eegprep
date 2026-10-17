# POP_LOADSET - Load an EEGLAB dataset

`pop_loadset` loads EEGLAB/EEGPrep `.set` datasets into an EEG dictionary.

Usage:

```python
EEG = pop_loadset("sample.set")
metadata = pop_loadset("sample.set", loadmode="info")
channel_10 = pop_loadset("sample.set", loadmode=10)
```

`loadmode="info"` loads the dataset metadata while leaving `EEG["data"]` as
the stored data filename when that sidecar exists, or `"in set file"` when the
samples are stored in the `.set`. A numeric array already stored in the `.set`
is loaded even if `datfile` names a missing `.fdt`. An integer or sequence loads
only those channels using EEGLAB's 1-based channel numbers and clears the ICA
fields, which no longer describe the selected channel matrix.

A length-1 channel, sample, or trial axis is kept. A one-channel recording
reloads as `(1, pnts)`, and a one-component or one-channel ICA matrix stays
two-dimensional, including a 1-by-1 sphere. A multi-dimensional cell or
struct array keeps its MATLAB shape. Empty arrays keep the dimensions stored
in the file, so `[]` stays 0-by-0. A saved dataset stores `icaact` as `[]`.
Continuous `icaact` reloads as `(ncomp, pnts)` after it is recomputed from
the decomposition and the data. Event and channel indices load as integers,
and integral `run` and `session` labels stay integers.

When `EEG_OPTIONS["option_memmapdata"] = 1`, two-file datasets saved with an
`.fdt` sidecar load through a NumPy-compatible memory map. Single-file `.set`
datasets still load into memory.

The main-window "Load existing dataset" action stores the loaded dataset in
the shared `EEGPrepSession`, updates `EEG`, `ALLEEG`, and `CURRENTSET`, and
records the load command in history.

See also: POP_SAVESET, LOADSET
