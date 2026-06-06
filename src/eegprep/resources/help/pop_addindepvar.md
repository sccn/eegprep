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

EEGPrep supports the command-line design-selection behavior used by
`pop_studydesign` and `std_makedesign`. MATLAB GUI callback strings and direct
figure mutation are not available in standalone EEGPrep.

See also: POP_LISTFACTORS, POP_STUDYDESIGN, STD_MAKEDESIGN
