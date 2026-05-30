# POP_DELSET - Delete datasets from ALLEEG

`pop_delset` removes one or more datasets from the loaded `ALLEEG` list.

Usage:

```python
ALLEEG, com = pop_delset(ALLEEG, 2)
ALLEEG, com = pop_delset(ALLEEG, [1, 3])
```

Dataset indices are EEGLAB-style 1-based values. The main-window
"Clear dataset(s)" and "Delete dataset(s) from memory" actions use the shared
session helpers so `EEG`, `ALLEEG`, and `CURRENTSET` stay synchronized with the
console.

See also: EEG_STORE, EEG_RETRIEVE
