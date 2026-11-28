# POP_FILEIO_MFF - Import EGI MFF recordings

Choose File > Import data > Using EEGPrep functions and plugins > Import
Magstim/EGI .mff file, then select the recording's `.mff` folder. The menu
calls `pop_mffimport`, which preserves the epochs in segmented recordings.

```python
EEG = pop_mffimport("recording.mff")
```

See also: POP_MFFIMPORT, POP_FILEIO
