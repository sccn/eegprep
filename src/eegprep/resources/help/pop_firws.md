# pop_firws

Filter an EEG dataset using a windowed-sinc FIR filter from the bundled firfilt
plugin.

```python
EEG = pop_firws(EEG, fcutoff=[1, 40], ftype="bandpass", forder=330)
EEG = pop_firws(EEG, fcutoff=1, ftype="highpass", forder=330)
```

Supported windows are `rectangular`, `hann`, `hamming`, `blackman`, and
`kaiser`. Boundary events split continuous data before filtering, and ICA
activations are cleared after filtering.
