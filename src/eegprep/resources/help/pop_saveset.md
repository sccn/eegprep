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
sidecar. Resave only writes a dataset that already has a `.set` filename.
A dataset imported from `.vhdr`, `.edf`, or any other raw file has not been
saved yet: pass a `.set` path, or use File > Save current dataset as. Resave
will not replace the original recording. Any other filename is stored as
`.set`, including a name whose extension is not `.set`.

A plain save follows the current `option_savetwofiles` setting and
may inline data into the `.set` file when that option is disabled.
A missing parent folder is created. If the path still cannot be written,
`pop_saveset` raises an error and leaves the dataset unsaved.

The saved dataset stores `icaact` as `[]`. The dataset kept in memory still
has its activations, and loading the file recomputes them.

The main-window save actions update dataset `filename`, `filepath`, and
`saved` metadata and record the save command in session history. File >
Resave current dataset(s) on a never-saved dataset opens Save as instead of
writing over the source file. Resaving multiple selected datasets requires
each dataset to already have a `.set` filename.

See also: POP_LOADSET
