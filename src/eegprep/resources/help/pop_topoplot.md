# pop_topoplot

`pop_topoplot` plots EEGLAB-style 2-D scalp maps.

Use `typeplot=1` to plot channel ERP maps at selected latencies in
milliseconds:

```python
pop_topoplot(EEG, typeplot=1, items=[0, 100, 200], topotitle="ERP maps")
```

Use `typeplot=0` to plot ICA component maps. Component numbers are 1-based to
match EEGLAB. Negative component numbers invert polarity and `float("nan")`
leaves a blank subplot:

```python
pop_topoplot(EEG, typeplot=0, items=[1, -2, float("nan"), 3], topotitle="IC maps")
```

Additional `topoplot` options can be passed as keyword arguments, for example
`electrodes="on"`, `colorbar="off"`, or `maplimits=[-5, 5]`.

DIPFIT dipole overlays and 3-D head plots are handled by later Phase 4 work.

When scripting, pass `plot='off'` to build the figure without opening a window; the default `plot='on'` displays it.
