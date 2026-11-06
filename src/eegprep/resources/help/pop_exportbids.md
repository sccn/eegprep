# POP_EXPORTBIDS - Export EEG datasets to BIDS

`pop_exportbids` writes one or more EEGPrep datasets to a minimal BIDS EEG
folder with EEGLAB `.set` files and sidecar tables.

Usage:

```python
output_dir = pop_exportbids(EEG, "bids_out")
output_dir, com = pop_exportbids([EEG1, EEG2], "bids_out", return_com=True)
```

The exporter writes `dataset_description.json`, `README`, `participants.tsv`,
`*_eeg.json`, channel tables, electrode tables, event tables, and EEG dataset
files. Event tables have `onset`, `duration`, `sample`, and `value`. A numeric
event type is written as `1`, not `1.0`. The electrode file name keeps the
`task-` entity.

`pop_exportbids(EEG, folder)` writes that dataset from memory. File > Export
to BIDS requires a loaded study and opens the output-folder dialog. A task
name cannot contain a space or an underscore.

See also: POP_IMPORTBIDS, BIDS_PREPROC
