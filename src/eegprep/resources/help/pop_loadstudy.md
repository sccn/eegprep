# POP_LOADSTUDY - Load an EEGPrep STUDY

`pop_loadstudy` loads a `.study` file saved by `pop_savestudy` or by EEGLAB.

EEGPrep saves JSON. EEGLAB saves a MATLAB file with a `STUDY` variable.
MATLAB character arrays are decoded as Latin-1, so characters such as `ä`
and `¸` survive. When the STUDY directory has moved, dataset paths that
lived under the saved study directory are rebased onto the new directory.
Other locations are tried beside the STUDY file and under the dataset's
parent folder names. `option_storedisk` selects a header-only load or a
full data load.

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
