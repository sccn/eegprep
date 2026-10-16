# POP_BIOSIG - Import BIOSIG-style files

`pop_biosig` imports EDF, BDF, and GDF files through EEGPrep's Python File-IO
path.

Usage:

```python
EEG = pop_biosig("recording.edf")
EEG, com = pop_biosig("recording.bdf", return_com=True)
EEG = pop_biosig("recording.edf", blockrange=[30, 60])
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
