# pop_studydesign

Create, edit, or select STUDY designs.

Design variables are read from `STUDY.datasetinfo` and from per-trial
`trialinfo` entries when available. Typical variables include `condition`,
`group`, `session`, `run`, and `subject`.

In the design editor, **New** under "Edit the independent variables for this
design" opens **Add variable**. A trial field such as reaction time (`rt`) can
be added as a continuous variable. Pressing OK stores it on `STUDY.design`.
When that field is not already in the STUDY trial metadata, EEGPrep rebuilds
`trialinfo` from the time-locking events before updating the design. Opening
the dialogs and canceling leaves the STUDY unchanged. Subject selection,
design New/Rename/Delete, Edit, Delete variable, and List factors are still
unavailable.

Example:

```python
STUDY, ALLEEG = pop_studydesign(
    STUDY,
    ALLEEG,
    1,
    variable1="condition",
    values1=["target", "standard"],
)
```

Measure precompute, plotting, LIMO, and clustering are handled by later STUDY
phases.
