# pop_chancenter

Recenter Cartesian channel coordinates.

`pop_chancenter(EEG, center, omitchans)` subtracts a supplied or optimized
sphere center from `EEG["chanlocs"]`, updates derived coordinate fields, and
returns the updated EEG structure. Channel indices in `omitchans` are
EEGLAB-facing 1-based indices.

The command-line path supports `return_com=True` so GUI and console history can
store a replayable command. The lightweight GUI entry point currently behaves as
a cancel path unless a renderer is supplied by a future dialog implementation.

Example:

```python
EEG, com = pop_chancenter(EEG, [0, 0, 0], [129], return_com=True)
```
