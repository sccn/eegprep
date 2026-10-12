# POP_ADDINDEPVAR - Select a STUDY independent variable

`pop_addindepvar` returns the variable name, selected values, and categorical
flag used by STUDY design helpers.

```python
variable, values, categorical = pop_addindepvar(
    STUDY,
    var="condition",
    values=["target", "standard"],
    vartype="categorical",
)
```

Pass `gui=True`, or use the study design editor's **New** button, to open
**Add variable**. The dialog lists available factors, including continuous
trial fields such as `rt`, and can combine selected categorical values.
Command-line selection with `var` and `values` remains available.
MATLAB callback strings and figure handles still raise `NotImplementedError`.

See also: POP_LISTFACTORS, POP_STUDYDESIGN, STD_MAKEDESIGN
