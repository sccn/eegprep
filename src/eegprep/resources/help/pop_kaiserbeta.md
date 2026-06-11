# pop_kaiserbeta

Estimate the Kaiser-window beta parameter from a requested maximum passband
deviation.

```python
beta = pop_kaiserbeta(0.001)
beta, command = pop_kaiserbeta(0.001, return_com=True)
```

Calling `pop_kaiserbeta()` without arguments opens the EEGLAB-style helper
dialog. The returned history command is replayable in `eegprep-console`.

Use this with `pop_firwsord(..., wtype="kaiser", dev=...)` and `pop_firws` or
`pop_xfirws` when designing Kaiser-window FIR filters.
