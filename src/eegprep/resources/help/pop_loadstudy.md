# POP_LOADSTUDY - Load an EEGPrep STUDY

`pop_loadstudy` loads an EEGPrep `.study` JSON file.

Usage:

```python
STUDY, ALLEEG, com = pop_loadstudy("analysis.study")
```

The loaded STUDY receives `filename` and `filepath` fields. On the main
window, loading a STUDY sets `CURRENTSTUDY` to 1 and synchronizes the console
workspace.

This is the standalone EEGPrep STUDY format currently implemented on this
branch.

See also: POP_SAVESTUDY, POP_STUDY
