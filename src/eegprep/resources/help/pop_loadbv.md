# POP_LOADBV - Import a BrainVision recording

`pop_loadbv` imports a BrainVision Data Exchange recording from its `.vhdr`
header. The companion data (`.eeg` or `.dat`) and optional marker (`.vmrk`)
files must remain beside the header, using the filenames recorded in it.

```python
from eegprep import pop_loadbv

EEG = pop_loadbv("record.vhdr")
EEG = pop_loadbv("/data/session", "subject01.vhdr")
EEG = pop_loadbv("/data/session", "subject01.vhdr", [1001, 5000], [1, 2, 8])
```

Sample ranges and channel indices are 1-based, and a two-value sample range is
inclusive. A scalar sample selects from that sample through the end. Set
`metadata=True` to read dimensions, channel information, and markers without
loading signal values.

Binary `INT_16`, `UINT_16`, and `IEEE_FLOAT_32` recordings are supported in
multiplexed and vectorized orientation. Multiplexed and vectorized ASCII data
are also supported. Voltage channels are returned in microvolts; each channel's
original BrainVision unit and resolution are preserved in `bvunit` and
`bvresolution`.

Marker positions and durations are measured in samples. Event latency remains
1-based, while `urevent` pointers use EEGPrep's 0-based internal convention.
Marker-based or fixed-time recordings with uniformly spaced `New Segment`
markers are returned as channel-by-sample-by-trial arrays.
Malformed headers, invalid selections, and truncated binary data raise a clear
error rather than returning an inconsistent dataset.
