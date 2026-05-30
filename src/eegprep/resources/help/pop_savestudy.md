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
`saved="yes"` fields. Phase 5b cached measure arrays in `STUDY.changrp` and
`STUDY.cluster` are saved in the `.study` JSON file using EEGPrep-owned
structured data, not EEGLAB sidecar files.

See also: POP_LOADSTUDY, POP_STUDY
