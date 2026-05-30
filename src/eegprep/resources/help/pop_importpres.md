# POP_IMPORTPRES - Import Presentation LOG events

`pop_importpres` imports Presentation `.LOG` event files through
`pop_importevent`.

Usage:

```python
EEG = pop_importpres(EEG, "experiment.log")
EEG, com = pop_importpres(EEG, "experiment.log", return_com=True)
```

The default fields are event type and latency. Pass explicit `fields` or
`timeunit` options when your log has a different structure.

See also: POP_IMPORTEVENT
