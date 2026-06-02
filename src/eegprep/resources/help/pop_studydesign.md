# pop_studydesign

Create, edit, or select STUDY designs.

Design variables are read from `STUDY.datasetinfo` and from per-trial
`trialinfo` entries when available. Typical variables include `condition`,
`group`, `session`, `run`, and `subject`.

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
