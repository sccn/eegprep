# pop_envtopo

Plots data envelopes and the largest ICA component projection envelopes.

```python
fig, com = pop_envtopo(EEG, timerange=[-100, 300], return_com=True)
```

This requires epoched data, channel locations, and ICA weights/maps.
