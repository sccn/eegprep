# pop_firma

Filter an EEG dataset using a moving-average FIR filter from the bundled firfilt
plugin.

```python
EEG = pop_firma(EEG, forder=10)
```

Boundary events split continuous data before filtering, and ICA activations are
cleared after filtering.
