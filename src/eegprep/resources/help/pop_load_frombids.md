# POP_LOAD_FROMBIDS - Load one EEG file from a BIDS dataset

`pop_load_frombids` loads a supported EEG recording from a BIDS dataset and
applies BIDS metadata, channel, and event information when available.

Usage:

```python
EEG = pop_load_frombids("sub-01/eeg/sub-01_task-rest_eeg.set")
EEG, report = pop_load_frombids(
    "sub-01/eeg/sub-01_task-rest_eeg.set",
    return_report=True,
)
```

Supported data files include EEGLAB `.set`, EDF, BDF, and BrainVision `.vhdr`
files when the required readers are installed. `bidsevent` can be `"replace"`,
`"merge"`, `"append"`, or disabled. `infer_locations` can infer channel
locations from known channel labels or a packaged montage resource.

Use `pop_importbids` or `bids_preproc` when you need to scan or process a whole
BIDS dataset.

See also: POP_IMPORTBIDS, POP_EXPORTBIDS, BIDS_PREPROC
