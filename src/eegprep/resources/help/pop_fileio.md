# POP_FILEIO - Import EEG files through Python File-IO readers

`pop_fileio` imports supported EEG file formats using EEGPrep's Python reader
stack.

Supported paths include EEGLAB `.set`, MATLAB arrays, text/NumPy arrays,
EDF/BDF/GDF, BrainVision `.vhdr`/`.ahdr`, EGI `.mff`, Neuroscan `.cnt`, and
`.eeg` files.

For `.eeg` or `.EEG`, a sibling header with the same stem and a `.vhdr`
extension (or `.ahdr` if no `.vhdr` exists) selects BrainVision import.
Without either header, the file uses the native Neuroscan epoch reader.
The case of the data-file extension does not determine the format.

Usage:

```python
EEG = pop_fileio("recording.vhdr")
EEG = pop_fileio("recording.eeg")  # Uses sibling recording.vhdr or recording.ahdr.
EEG, com = pop_fileio("recording.edf", return_com=True)
EEG = pop_fileio("recording.bdf", blockrange=[10, 20])
EEG = pop_fileio("epochs.set", channels="1:16", trials=[2, 50])
EEG = pop_fileio("epochs.set", "channels", "1:16", "trials", [2, 50])
```

For BrainVision and MNE-backed continuous formats, `blockrange=[start, stop]`
imports a half-open interval measured in seconds. BrainVision always uses
`pop_loadbv`, including `.ahdr` headers and time-range imports, without requiring
MNE. Time bounds round to the nearest sample using the header's sampling rate;
the stop is clipped to the recording length. When combined with `blockrange`,
`samples` selects within that time window. BrainVision data-file paths use the
same options as their header paths; history retains the filename supplied by
the caller.

`channels` accepts 1-based channel numbers. `samples=[start, stop]` and
`trials=[start, stop]` are inclusive, 1-based ranges matching File-IO and
EEGLAB history commands. Selections update dataset dimensions, events, and ICA
metadata consistently.

Use the more specific File menu entries when you want format-specific file
filters in the GUI.

See also: POP_BIOSIG, POP_IMPORTDATA, POP_LOADSET

Use `gui=True` to select channels and inclusive sample/trial ranges with
1-based indices. EGI `.raw` uses `pop_readegi`. MFF recordings are directories;
select the `.mff` folder itself and install `pip install "eegprep[mne]"`.
