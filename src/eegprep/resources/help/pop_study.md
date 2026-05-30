# POP_STUDY - Create a STUDY

`pop_study` creates a minimal EEGPrep STUDY structure from loaded `ALLEEG`
datasets.

Usage:

```python
STUDY, ALLEEG, com = pop_study(None, ALLEEG, name="My study")
```

The STUDY records dataset information such as set name, filename, path,
subject, condition, session, and group. On the main window, creating a STUDY
sets `CURRENTSTUDY` to 1 and keeps `STUDY`, `ALLEEG`, and console state in the
shared session.

Design editing, precompute, and clustering surfaces remain Phase 5 work until
their branches land.

See also: POP_STUDYWIZARD, POP_STUDYERP, POP_SAVESTUDY
