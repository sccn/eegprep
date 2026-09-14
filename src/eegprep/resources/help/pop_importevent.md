# POP_IMPORTEVENT - Import event information

`pop_importevent` imports event records from a text table or record sequence.

Usage:

```python
EEG = pop_importevent(EEG, "event", "events.tsv", "append", "no")
EEG, com = pop_importevent(EEG, "event", records, return_com=True)
```

Imported events are normalized into EEGLAB-style event dictionaries with
1-based sample latencies. Existing events are appended by default; use
`"append", "no"` to replace them. `timeunit` gives the latency unit in seconds,
so millisecond tables use `1e-3`; use `numpy.nan` when values are already sample
positions. Duration values use the same unit and are converted to sample counts.

`align` aligns an imported event to an existing event: `0` aligns the first
events, a positive value selects that zero-based offset in the existing table,
and a negative value selects a later imported event to align to the first
existing event. `optimalign="on"` additionally estimates a sampling-rate ratio
when alignment is requested. `indices` contains 1-based event indices to update
instead of appending rows. Use `delim` for a delimiter that cannot be inferred
from the filename, for example `"delim", ","` for comma-separated `.txt` data.

See also: POP_CHANEVENT, POP_IMPORTPRES, POP_IMPORTERPLAB
