# POP_EXPORTBIDS - Export EEG datasets to BIDS

`pop_exportbids` writes one or more EEGPrep datasets to a minimal BIDS EEG
folder with EEGLAB `.set` files and sidecar tables.

Usage:

```python
output_dir = pop_exportbids(EEG, "bids_out")
output_dir, com = pop_exportbids([EEG1, EEG2], "bids_out", return_com=True)
```

The exporter writes `dataset_description.json`, `README`, `participants.tsv`,
`*_eeg.json`, channel tables, electrode tables, `*_coordsystem.json`, event
tables, and EEG dataset files. Event tables have `onset`, `duration`, `sample`,
and `value`. A numeric event type is written as `1`, not `1.0`. Electrode and
coordinate-system file names omit `task-` and `desc-`.

`pop_exportbids(EEG, folder)` writes that dataset from memory. File > Export
to BIDS requires a loaded study and opens the output-folder dialog. Each
dataset is written under `EEG.subject`, or `STUDY.datasetinfo` subject when
the dataset subject is empty. A dataset with neither set raises an error that
names `pop_exportbids` and tells you to add the subject. One dataset with no
subject still uses the `subject` argument. `createids=True` (the anonymized-ID
checkbox) assigns `sub-001`, `sub-002`, and so on, one id per subject, and
`participants.tsv` has one row per subject. Session and run are taken from
the dataset when they are set. A task name cannot contain a space or an
underscore.

The dialog records a command that includes the dataset list, task, license,
changes, and createids, so running it in `eegprep-console` repeats the export.
EEGLAB leaves this command empty (`pop_exportbids.m` lines 247-252).

See also: POP_IMPORTBIDS, BIDS_PREPROC
