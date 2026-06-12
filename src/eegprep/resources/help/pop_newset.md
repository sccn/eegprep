# pop_newset

`pop_newset` stores, retrieves, names, and switches EEG datasets in the
EEGPrep `ALLEEG` list.

From the GUI, EEGPrep uses this dialog after data-changing actions such as
resampling, filtering, epoching, selecting data, rereferencing, interpolation,
and cleaning. Choose whether the processed dataset should overwrite the current
dataset or be stored as a new dataset.

Use **Edit description** to open a multiline editor for the dataset
`comments` field.

Common command-line forms:

```python
ALLEEG, EEG, CURRENTSET, LASTCOM = pop_newset(ALLEEG, EEG, CURRENTSET, "overwrite", "on")
ALLEEG, EEG, CURRENTSET, LASTCOM = pop_newset(ALLEEG, EEG, CURRENTSET, "setname", "cleaned")
ALLEEG, EEG, CURRENTSET, LASTCOM = pop_newset(ALLEEG, EEG, CURRENTSET, "retrieve", 2)
```

EEGPrep uses EEGLAB-facing 1-based dataset indices for `CURRENTSET`. Python
array indices inside EEG data remain 0-based.
