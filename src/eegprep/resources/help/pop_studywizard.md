# POP_STUDYWIZARD - Create a STUDY by browsing datasets

`pop_studywizard` loads selected dataset files and creates an EEGPrep STUDY.

Usage:

```python
STUDY, ALLEEG, com = pop_studywizard(["sub-01.set", "sub-02.set"])
```

The main-window action prompts for one or more EEGPrep/EEGLAB dataset files,
loads them with `pop_loadset`, and then calls `pop_study`.

See also: POP_STUDY, POP_LOADSET
