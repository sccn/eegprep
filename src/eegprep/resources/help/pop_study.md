# pop_study

Create or edit a STUDY structure from datasets already loaded in `ALLEEG`.

`pop_study` records each dataset's file path, subject, condition, group,
session, run, component list, and available trial metadata. Dataset numbers in
the STUDY are 1-based to match EEGLAB.

Example:

```python
STUDY, ALLEEG = pop_study(STUDY, ALLEEG, name="Oddball")
```

Use `pop_studywizard` to browse for dataset files before creating a STUDY.
