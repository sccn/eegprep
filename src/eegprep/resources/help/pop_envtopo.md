# pop_envtopo

Plots data envelopes and the largest ICA component projection envelopes.

```python
fig, com = pop_envtopo(EEG, timerange=[-100, 300], return_com=True)
```

This requires epoched data, channel locations, and ICA weights/maps.

EEGPrep's standalone wrapper accepts one dataset. Multi-dataset envelope
comparison is not implemented because component maps, ICA channel subsets, and
dataset-level envelopes need a dedicated group workflow rather than a silent
merge.
