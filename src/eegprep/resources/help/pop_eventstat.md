POP_EVENTSTAT - Plot statistics for numeric event fields.

`pop_eventstat` extracts numeric values from `EEG["event"]`, optionally filters
by event type and latency range, and runs `signalstat` on the resulting event
values.

Examples:

```python
stats = pop_eventstat(EEG, "latency", [], [], 5)
stats = pop_eventstat(EEG, "duration", ["square"], [0, 300], 5)
```

String-valued event fields are ignored because the statistical workflow requires
numeric values.

When scripting, pass `plot='off'` to build the figure without opening a window; the default `plot='on'` displays it.
