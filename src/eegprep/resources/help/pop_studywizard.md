# POP_STUDYWIZARD - Create a STUDY by browsing datasets

`pop_studywizard` loads selected dataset files and creates a STUDY from them.

Usage:

```python
STUDY, ALLEEG, com = pop_studywizard(
    ["sub-01.set", "sub-02.set"],
    name="Oddball",
    return_com=True,
)
```

After loading the datasets with `pop_loadset`, the helper calls `pop_study` to
build STUDY metadata. Use `pop_study` to edit dataset metadata or
`pop_studydesign` to edit design variables after creation.

See also: POP_STUDY, POP_LOADSET, POP_STUDYDESIGN
