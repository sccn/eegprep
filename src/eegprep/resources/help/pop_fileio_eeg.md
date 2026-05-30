# POP_FILEIO_EEG - Import EEG files

The EEG File-IO menu action imports compatible `.eeg` recordings through
`pop_fileio`.

Usage:

```python
EEG = pop_fileio("recording.eeg")
```

For BrainVision recordings, selecting the `.vhdr` header is usually preferred
because it references the data and marker files explicitly.

See also: POP_FILEIO, POP_FILEIO_BRAINVISION
