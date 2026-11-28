# POP_IMPORTBIDS - Import BIDS EEG data

`pop_importbids` loads one or more supported EEG files from a BIDS dataset or a
single BIDS EEG file.

Usage:

```python
EEG = pop_importbids("sub-01/eeg/sub-01_task-rest_eeg.set")
STUDY, ALLEEG = pop_importbids("bids_root")
STUDY, ALLEEG, com = pop_importbids("bids_root", return_com=True)
```

A single file returns that dataset. A directory returns `(STUDY, ALLEEG)`:
EEGPrep scans for supported EEG files, copies each dataset into
`derivatives/eeglab` (or `outputdir`), and writes a `.study` file there. The
raw BIDS folder is not modified. The File menu import shows the options
dialog first: study name, channel locations, the events column used as the
event type, and optional task, session, run, and subject filters. The event
type popup lists fields from visible `events.tsv` files with `value` first.
It excludes onset, duration, sample, stim_file and response_time. Without an
event type field, BIDS event import starts unchecked and the popup shows `n/a`. Those task,
session, run, and recording choices are a popup and listboxes filled by
scanning the folder. "Import only the following participant indices" takes
1-based positions in `participants.tsv`, or in the sorted `sub-*` folders
when that file is missing. `2` selects the second participant. A value that
is not a positive whole number, or that is past the end of the list, raises
an error that names `pop_importbids` and shows the valid range. The history
command records the numbers, for example `'subjects', [1 2]`. A folder import
records `[STUDY ALLEEG] = pop_importbids(...)` and does not pass
`return_study`. It includes the selected recordings, task, sessions, and
runs. Running that command again rebuilds the same study. Passing
`subjects=[1, 2]` to `pop_importbids` uses those same 1-based indices, so the
recorded command imports the same participants. An index outside the list
raises an error that names `pop_importbids` and the valid range. The event
type defaults to the `value` column. `trial_type` is used when `value` is
absent.

EDF and BDF files are read with `pop_biosig`. TSV and JSON sidecars are decoded
as UTF-8, and a leading UTF-8 BOM is ignored. A TSV that is not valid UTF-8 is
decoded as Latin-1, so a `0xB5` units byte stays µ. A dataset imported this way
can be saved with `pop_saveset`.

See also: POP_EXPORTBIDS, POP_LOAD_FROMBIDS
