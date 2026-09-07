# pop_envtopo

Plots data envelopes and the largest ICA component projection envelopes.

```python
fig, com = pop_envtopo(EEG, timerange=[-100, 300], return_com=True)
```

This requires epoched data, channel locations, and ICA weights/maps.

Leaving the "Component numbers to remove from data before plotting" field blank
removes no components.

Left-click the envelope panel or any scalp map to enlarge it in a pop-up window;
each enlarged map is annotated with its ranking metric value.

EEGPrep's standalone wrapper accepts one dataset. Multi-dataset envelope
comparison is not implemented because component maps, ICA channel subsets, and
dataset-level envelopes need a dedicated group workflow rather than a silent
merge.

When scripting, pass `plot='off'` to build the figure without opening a window; the default `plot='on'` displays it.
