# pop_firpm

Filter an EEG dataset using a Parks-McClellan equiripple FIR filter from the
bundled firfilt plugin.

```python
EEG = pop_firpm(EEG, fcutoff=[1, 40], ftrans=1, ftype="bandpass", forder=330)
```

The implementation uses SciPy's `remez` routine. Boundary events split
continuous data before filtering, and ICA activations are cleared after
filtering.
