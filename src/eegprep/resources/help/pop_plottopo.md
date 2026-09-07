# pop_plottopo

Plots channel ERP traces in a scalp-positioned array by default, or in a
rectangular grid when `rect=True`.

```python
fig, com = pop_plottopo(EEG, chans=[1, 2, 3], rect=False, return_com=True)
```

Channel indices are EEGLAB-facing and 1-based.

When scripting, pass `plot='off'` to build the figure without opening a window; the default `plot='on'` displays it.
