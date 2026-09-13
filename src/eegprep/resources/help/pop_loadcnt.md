# POP_LOADCNT - Import a Neuroscan CNT recording

`pop_loadcnt` reads continuous Neuroscan `.cnt` recordings into an EEGPrep
dataset. Data are returned in microvolts with channel-major shape
`(channels, samples)`, and event latencies are 1-based like other EEGPrep
events.

```python
from eegprep import pop_loadcnt

EEG = pop_loadcnt("recording.cnt", dataformat="auto")
```

Use `sample1` for a zero-based starting sample or `t1` for a starting time in
seconds. `ldnsamples` selects an exact number of samples and takes precedence
over the duration in `lddur`.

```python
EEG = pop_loadcnt(
    "recording.cnt",
    dataformat="int32",
    sample1=5_000,
    ldnsamples=10_000,
    keystroke="on",
)
```

Set `scale="off"` to obtain stored integer counts represented in the requested
floating-point `precision`. For recordings too large to keep in memory, pass a
`.fdt` path as `memmapfile`; `EEG["data"]` will be a disk-backed array.

CNT is also used by ANT Neuro, but that is a different file format. This loader
supports Neuroscan CNT files only.

See also: `loadcnt`, `pop_fileio`
