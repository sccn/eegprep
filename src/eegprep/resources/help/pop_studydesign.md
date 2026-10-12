# pop_studydesign

Create, edit, or select STUDY designs.

Design variables are read from `STUDY.datasetinfo` and from per-trial
`trialinfo` entries when available. Typical variables include `condition`,
`group`, `session`, `run`, and `subject`.

In the design editor:

- The subject box and the `...` button choose the subjects stored on every
  design. An empty box means all subjects. Unknown names are rejected.
- **New**, **Rename**, and **Delete** under Design name copy, rename, or remove
  a design. The first design cannot be deleted.
- **New**, **Edit**, and **Delete** under the independent-variable list add,
  replace, or remove a factor. **New** and **Edit** open **Add variable**.
  A trial field such as reaction time (`rt`) can be continuous.
- **List factors** shows the first-level and second-level factor text for the
  selected design.
- **Re-save STUDY file** writes the STUDY again when it already has a file name.

Pressing OK updates each changed design, selects the highlighted design, and
resaves when that box is checked. When a new factor is not already in the
STUDY trial metadata, EEGPrep rebuilds `trialinfo` from the time-locking
events first. Cancel leaves the STUDY unchanged. `std_makedesign` stores the
first four independent variables.

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
