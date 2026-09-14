# POP_IMPORTPRES - Import Presentation LOG events

`pop_importpres` imports Presentation `.LOG` event files through
`pop_importevent`.

Usage:

```python
EEG = pop_importpres(EEG, "experiment.log")
EEG, com = pop_importpres(EEG, "experiment.log", return_com=True)
EEG = pop_importpres(EEG, "experiment.log", "Event Type", "Time", "Duration")
```

For tab-delimited Presentation logs, EEGPrep finds the header row, uses `code`
and `time` by default, and converts Presentation's 0.1 ms timestamps to EEG
samples. The positional field names select another event-type, latency, or
duration column. Pass `timeunit` to override the timestamp unit. Simple event
tables without a Presentation header continue to use `fields` and `timeunit`
as accepted by `pop_importevent`.

See also: POP_IMPORTEVENT
