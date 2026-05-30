# POP_LOADSET - Load an EEGLAB dataset

`pop_loadset` loads EEGLAB/EEGPrep `.set` datasets into an EEG dictionary.

Usage:

```python
EEG = pop_loadset("sample.set")
```

The main-window "Load existing dataset" action stores the loaded dataset in
the shared `EEGPrepSession`, updates `EEG`, `ALLEEG`, and `CURRENTSET`, and
records the load command in history.

See also: POP_SAVESET, LOADSET
