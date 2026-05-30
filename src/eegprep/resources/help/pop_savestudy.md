# POP_SAVESTUDY - Save an EEGPrep STUDY

`pop_savestudy` saves the current STUDY dictionary as an EEGPrep-owned
`.study` JSON file.

Usage:

```python
STUDY, com = pop_savestudy(
    STUDY,
    ALLEEG,
    filename="demo.study",
    filepath="/data",
    return_com=True,
)
STUDY, com = pop_savestudy(STUDY, ALLEEG, savemode="resave", return_com=True)
```

The saved file contains STUDY metadata, dataset membership, designs, and
consistency diagnostics. The saved STUDY receives `filename`, `filepath`, and
`saved="yes"` fields. It does not save measure precompute arrays or cluster
measure data in Phase 5a.

See also: POP_LOADSTUDY, POP_STUDY
