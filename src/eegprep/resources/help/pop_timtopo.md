# pop_timtopo

Plots channel ERP traces together with scalp maps at selected latencies.

```python
fig, com = pop_timtopo(EEG, plottimes=[100, 200], return_com=True)
```

If no latency is supplied, EEGPrep uses the latency with largest channel
variance, matching the common EEGLAB default.
