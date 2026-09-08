# POP_DELSET - Delete datasets from ALLEEG

`pop_delset` deletes one or more datasets from the loaded `ALLEEG` list.

Usage:

```python
ALLEEG, com = pop_delset(ALLEEG, 2)
ALLEEG, com = pop_delset(ALLEEG, [1, 3])
```

Dataset indices are EEGLAB-style 1-based values.

Deleting empties the dataset's slot in place instead of shifting the later
datasets down, so the remaining dataset numbers stay valid in the Datasets menu,
in `CURRENTSET`, and in recorded history commands. Deleting dataset 2 of three
leaves datasets 1 and 3, and an empty slot at 2. Trailing empty slots are
dropped, and the next dataset you create fills the lowest empty slot. Asking for
a dataset number that does not exist raises an error.

The main-window "Clear dataset(s)" and "Delete dataset(s) from memory" actions
use the shared session helpers so `EEG`, `ALLEEG`, and `CURRENTSET` stay
synchronized with the console. After a delete the selection moves to the nearest
remaining dataset.

See also: EEG_STORE, EEG_RETRIEVE
