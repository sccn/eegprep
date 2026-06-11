# POP_LISTFACTORS - List STUDY design factors

`pop_listfactors` returns EEGLAB-style factor descriptors for a STUDY or design
structure.

```python
factors = pop_listfactors(STUDY, level="both", vartype="both", constant="off")
```

Each descriptor includes the factor label, variable type, level, and value when
the factor is categorical. Use `constant="off"` to omit the intercept-like
constant factor.

The standalone EEGPrep helper does not open EEGLAB's MATLAB factor-list GUI.
Requests with `gui="on"` raise a clear `NotImplementedError`.

See also: POP_ADDINDEPVAR, POP_STUDYDESIGN, STD_ADDVARLEVEL
