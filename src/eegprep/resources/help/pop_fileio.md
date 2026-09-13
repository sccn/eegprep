# POP_FILEIO - Import EEG files through Python File-IO readers

`pop_fileio` imports supported EEG file formats using EEGPrep's Python reader
stack.

Supported paths include EEGLAB `.set`, MATLAB arrays, text/NumPy arrays,
EDF/BDF/GDF, BrainVision `.vhdr`, EGI `.mff`, Neuroscan `.cnt`, and compatible
`.eeg` files.

Usage:

```python
EEG = pop_fileio("recording.vhdr")
EEG, com = pop_fileio("recording.edf", return_com=True)
EEG = pop_fileio("recording.bdf", blockrange=[10, 20])
```

For MNE-backed continuous formats, `blockrange=[start, stop]` imports a
half-open interval measured in seconds.

Use the more specific File menu entries when you want format-specific file
filters in the GUI.

See also: POP_BIOSIG, POP_IMPORTDATA, POP_LOADSET
