# POP_SAVEH - Save command history

`pop_saveh` writes dataset or session history to a script file.

Usage:

```python
com = pop_saveh(ALLCOM, "eegprephist.m", ".")
```

The main-window History scripts menu can save the current dataset history or
the full session command list. Session history files include an
`eegprep.eeglab()` launcher line so users can reopen the GUI after running the
history script.

See also: POP_RUNSCRIPT, EEGH
