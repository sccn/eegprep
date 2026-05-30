# POP_LOADSTUDY - Load an EEGPrep STUDY

`pop_loadstudy` loads an EEGPrep-owned `.study` JSON file saved by
`pop_savestudy`.

Usage:

```python
STUDY, ALLEEG, com = pop_loadstudy(
    filename="demo.study",
    filepath="/data",
    return_com=True,
)
```

When dataset files referenced by the STUDY are available, `pop_loadstudy`
loads them into `ALLEEG` and checks the resulting STUDY/dataset consistency.
The loaded STUDY stores `filename`, `filepath`, and
`STUDY["etc"]["oldfilepath"]` for diagnostics. If dataset files have moved,
call `pop_loadstudy(..., load_datasets=False)` and relink datasets later.

On the main window, loading a STUDY sets `CURRENTSTUDY` to 1 and synchronizes
the shared `eegprep-console` workspace.

See also: POP_SAVESTUDY, POP_STUDY
