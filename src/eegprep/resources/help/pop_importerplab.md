# POP_IMPORTERPLAB - Import ERPLAB event labels and bins

`pop_importerplab` maps an ERPLAB event list and bin definitions onto existing
events. It retains their types and latencies, adding `newtype`, `eventind`,
and contrast fields. Enable non-contrast bins to also add 0/1 membership fields.

Usage:

```python
EEG = pop_importerplab(EEG, "events.txt", "bins.txt", True)
EEG, com = pop_importerplab(EEG, gui=True, return_com=True)
```

The event-list columns are numeric code, quoted name, and numeric event index:

```text
11 "Left target" 1
12 "Right target" 2
```

Bin definitions refer to those indices; contrasts refer to simple bins:

```text
b1 = (1) label Left
b2 = (2) label Right
b3 = b1-b2 label Side contrast
```

Here `Side_contrast` becomes `Left`, `Right`, or `none of these` for each event.
Overlapping or missing contrast operands are reported and skipped. Every
existing event type must have exactly one event-list entry. Both files are
required. Use `pop_importevent` for ordinary type/latency text tables.
Cancel leaves the dataset and command history unchanged.

See also: POP_IMPORTEVENT
