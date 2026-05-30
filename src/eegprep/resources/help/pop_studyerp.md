# POP_STUDYERP - Create a simple ERP STUDY

`pop_studyerp` creates a STUDY marked with a simple ERP design from loaded
datasets.

Usage:

```python
STUDY, ALLEEG, com = pop_studyerp(ALLEEG, return_com=True)
```

This helper creates STUDY metadata named `Simple ERP STUDY` and an initial
`ERP` design. Precomputing or plotting ERP measures is handled by later STUDY
measure workflows.

See also: POP_STUDY, POP_STUDYDESIGN, POP_CHANPLOT
