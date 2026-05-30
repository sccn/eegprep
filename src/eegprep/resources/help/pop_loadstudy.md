# pop_loadstudy

Load an EEGPrep `.study` JSON file and, when referenced dataset files are
available, load those datasets into `ALLEEG`.

Example:

```python
STUDY, ALLEEG = pop_loadstudy(filename="demo.study", filepath="/data")
```

The loaded STUDY stores the current file name and path and keeps the previous
path in `STUDY["etc"]["oldfilepath"]` for diagnostics.
