# POP_IMPORTBIDS - Import BIDS EEG data

`pop_importbids` loads one or more supported EEG files from a BIDS dataset or a
single BIDS EEG file.

Usage:

```python
EEG = pop_importbids("sub-01/eeg/sub-01_task-rest_eeg.set")
ALLEEG, com = pop_importbids("bids_root", return_com=True)
```

When a directory is supplied, EEGPrep scans for supported EEG files and returns
a single EEG dictionary or a list of dictionaries depending on how many files
are found. Pass `return_study=True` to also get an in-memory study. The File
menu import does that after showing the options dialog: study name, channel
locations, the events column used as the event type, and optional task,
session, run, and subject filters. Those task, session, run, and recording
choices are a popup and listboxes filled by scanning the folder. "Import only
the following participant indices" takes 1-based positions in
`participants.tsv`, or in the sorted `sub-*` folders when that file is
missing. `2` selects the second participant. A value that is not a positive
whole number, or that is past the end of the list, raises an error that names
`pop_importbids` and shows the valid range. The history command records the
numbers, for example `'subjects', [1 2]`. A folder import opened from the
dialog records `[STUDY, ALLEEG] = pop_importbids(...)` with `return_study`
on, and includes the selected recordings, task, sessions, and runs. Running
that command again rebuilds the same study. Passing `subjects=[1, 2]` to
`pop_importbids` uses those same 1-based indices, so the recorded command
imports the same participants. An index outside the list raises an error that
names `pop_importbids` and the valid range. The event type defaults to the `value`
column. `trial_type` is used when `value` is absent. The import does not copy
datasets into `derivatives/eeglab`.

EDF and BDF files are read with `pop_biosig`. TSV and JSON sidecars are decoded
as UTF-8, and a leading UTF-8 BOM is ignored. A TSV that is not valid UTF-8 is
decoded as Latin-1, so a `0xB5` units byte stays µ. A dataset imported this way
can be saved with `pop_saveset`.

See also: POP_EXPORTBIDS, POP_LOAD_FROMBIDS
