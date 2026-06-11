# pop_fusechanrej

`pop_fusechanrej(ALLEEG)` keeps common channels across datasets that share the
same `subject` and `session` fields. This mirrors the EEGLAB cleanup step used
after channel rejection so repeated runs for the same participant keep a common
channel set.

Datasets without a `subject` value are returned unchanged because EEGPrep cannot
infer a safe grouping key.
