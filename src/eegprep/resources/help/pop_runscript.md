# POP_RUNSCRIPT - Run a history script

`pop_runscript` runs a Python history script in the supplied namespace.

Usage:

```python
com = pop_runscript("history.py", namespace)
```

The GUI provides the shared session namespace containing `EEG`, `ALLEEG`,
`CURRENTSET`, and `STUDY`. After the script runs, EEGPrep copies those names
back into the session and refreshes the GUI.

MATLAB `.m` and text scripts are recognized but not executed by EEGPrep.

See also: POP_SAVEH
