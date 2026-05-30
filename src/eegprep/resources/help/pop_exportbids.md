# POP_EXPORTBIDS - Export EEG datasets to BIDS

`pop_exportbids` writes one or more EEGPrep datasets to a minimal BIDS EEG
folder with EEGLAB `.set` files and sidecar tables.

Usage:

```python
output_dir = pop_exportbids(EEG, "bids_out")
output_dir, com = pop_exportbids([EEG1, EEG2], "bids_out", return_com=True)
```

The exporter writes `dataset_description.json`, `participants.tsv`, channel
tables, event tables, and EEG dataset files. It is intended as EEGPrep-owned
BIDS output, not a runtime dependency on the EEGLAB EEG-BIDS plugin.

See also: POP_IMPORTBIDS, BIDS_PREPROC
