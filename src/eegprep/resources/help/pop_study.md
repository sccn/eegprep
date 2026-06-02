# POP_STUDY - Create or edit a STUDY

`pop_study` creates or edits an EEGPrep STUDY structure from datasets already
loaded in `ALLEEG`.

Usage:

```python
STUDY, ALLEEG, com = pop_study(
    STUDY,
    ALLEEG,
    name="Oddball",
    task="auditory",
    notes="Initial STUDY",
    return_com=True,
)
```

The STUDY records each dataset's file path, subject, condition, group,
session, run, component list, and available trial metadata. Dataset numbers in
the STUDY are 1-based to match EEGLAB. The GUI edits metadata for already
loaded datasets; use `pop_studywizard` to browse for dataset files before
creating a STUDY.

On the main window, creating or editing a STUDY sets `CURRENTSTUDY` to 1 and
keeps `STUDY`, `ALLEEG`, and the `eegprep-console` workspace synchronized.

See also: POP_STUDYWIZARD, POP_STUDYERP, POP_STUDYDESIGN, POP_SAVESTUDY
