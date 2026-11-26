# POP_BIOSIG - Import BIOSIG-style files

`pop_biosig` imports EDF, BDF, and GDF files through EEGPrep's Python File-IO
path.

Usage:

```python
EEG = pop_biosig("recording.edf")
EEG, com = pop_biosig("recording.bdf", return_com=True)
EEG = pop_biosig("recording.edf", blockrange=[30, 60])
EEG = pop_biosig("recording.edf", "blockrange", [30, 60])
```

`blockrange=[start, stop]` loads a half-open time interval in seconds: the
sample at `start` is included and the sample at `stop` is not. The stop time is
clamped to the recording duration. Status decoding reads the BDF records that
overlap that interval.

BioSemi BDF import decodes the `Status` channel with `bdfeventmode`. The
default `4` stores a trigger at each rising edge of the lower 16 bits and
ignores samples whose epoch bit is set. Modes 1, 2, 3, 4, 5, 6, 7, 8, and 99
are accepted; any other value raises `ValueError`. Modes 3 and 5 keep each
trigger code on the edge that produced it. `importevent` defaults to `on` and
replaces file annotations with those triggers; `off` leaves the annotations in
place. `rmeventchan` defaults to `on` and removes `Status`; `off` keeps it.
Values other than `on` and `off` raise `ValueError`.

Use `pop_fileio` for other formats such as BrainVision, EGI MFF, CNT, or
EEGLAB `.set` files.

See also: POP_FILEIO, POP_IMPORTDATA, POP_LOADSET

Use `gui=True` for channel/range/event/reference options. References are 1-based indices in the loaded channels after Status removal. `importannot="off"` suppresses annotations independently of marker-event import. `memorymapped="on"` stores the imported samples in a temporary mapping. Install readers with `pip install "eegprep[mne]"`.

`uncalibrated="on"` returns the integers stored in an EDF or BDF file instead
of physical values. `overflow="on"` leaves EDF and BDF data unchanged, as
BIOSIG does for these two formats, and raises `ValueError` for GDF.
`blockepoch="off"` (the default) imports continuous data; `"on"` raises
`ValueError`.

The dialog has the controls of EEGLAB's dialog except two. "Import using
alternative BIOSIG method (mexSLOAD)" is left out because mexSLOAD is a
compiled BIOSIG reader and EEGPrep reads with MNE-Python. "Import as memory
mapped file" is left out because the Python reader loads the file into memory;
`memorymapped` stays a command-line option.
