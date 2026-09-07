# pop_timtopo

Plots channel ERP traces together with scalp maps at selected latencies.

```python
fig, com = pop_timtopo(EEG, plottimes=[100, 200], return_com=True)
```

If no latency is supplied, EEGPrep maps the latency of maximum field strength
(peak RMS across channels), matching the EEGLAB default.

When scripting, pass `plot='off'` to build the figure without opening a window; the default `plot='on'` displays it.
