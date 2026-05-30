# POP_SAVESTUDY - Save an EEGPrep STUDY

`pop_savestudy` saves the current STUDY dictionary as an EEGPrep `.study` JSON
file.

Usage:

```python
STUDY, com = pop_savestudy(STUDY, EEG, "analysis.study")
STUDY, com = pop_savestudy(STUDY, EEG, savemode="resave")
```

The saved STUDY receives `filename` and `filepath` fields. The main-window
save actions append the returned command to the shared session history.

See also: POP_LOADSTUDY, POP_STUDY
