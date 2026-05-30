# POP_FILEIO_BRAINVISION - Import BrainVision recordings

The BrainVision File-IO menu action imports `.vhdr` recordings through
`pop_fileio`.

Usage:

```python
EEG = pop_fileio("recording.vhdr")
```

Select the `.vhdr` header file. EEGPrep relies on the companion BrainVision
data and marker files referenced by the header.

See also: POP_FILEIO, POP_FILEIO_BRAINVISION_MAT
