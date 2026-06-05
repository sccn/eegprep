# POP_IMPORTGROUPVAR - Import a STUDY group variable

`pop_importgroupvar` attaches one value per STUDY subject and adds that
variable to the selected STUDY design.

```python
STUDY, com = pop_importgroupvar(
    STUDY,
    1,
    variable="age_group",
    values={"S01": "young", "S02": "older"},
    return_com=True,
)
```

Values may be a subject-to-value mapping, a sequence in design subject order,
or a text file with one value per subject.

See also: POP_STUDYDESIGN, POP_LISTFACTORS, STD_BUILDDESIGNMAT
