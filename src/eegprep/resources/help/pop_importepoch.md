# POP_IMPORTEPOCH - Import epoch metadata

`pop_importepoch` imports one row of metadata per epoch from a text table or a
two-dimensional Python sequence.

Usage:

```python
EEG = pop_importepoch(EEG, "epochs.tsv", ["condition", "rt"])
EEG, com = pop_importepoch(
    EEG,
    rows,
    ["epoch", "response", "rt"],
    "latencyfields", ["rt"],
    "typefield", "response",
    "timeunit", 1e-3,
    return_com=True,
)
```

The current dataset must be epoched, and the number of imported rows must match
`EEG["trials"]`. By default, prior events are cleared and a `TLE` event is
created at time zero in every epoch. `typefield` supplies those event types.
Every name in `latencyfields` creates another event per epoch; corresponding
`durationfields` values are optional and use the same `timeunit`. Event
latencies remain 1-based absolute sample positions, while `EEG["epoch"]` also
contains the imported row metadata and derived epoch-relative event fields.

Use `headerlines` to skip leading file rows and `clearevents="off"` to retain
existing epoched events. Existing continuous-style events without an `epoch`
field cannot be retained because they cannot be associated with imported rows.

See also: POP_IMPORTEVENT, POP_EPOCH
