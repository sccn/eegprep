# POP_BIOSIG - Import BIOSIG-style files

`pop_biosig` imports EDF, BDF, and GDF files through EEGPrep's Python File-IO
path.

Usage:

```python
EEG = pop_biosig("recording.edf")
EEG, com = pop_biosig("recording.bdf", return_com=True)
```

Use `pop_fileio` for other formats such as BrainVision, EGI MFF, CNT, or
EEGLAB `.set` files.

See also: POP_FILEIO, POP_IMPORTDATA, POP_LOADSET
