# pop_eegfiltnew

Filter an EEG dataset with EEGLAB's default Hamming-windowed sinc FIR workflow.

Common examples:

```python
EEG = pop_eegfiltnew(EEG, locutoff=1)
EEG = pop_eegfiltnew(EEG, hicutoff=40)
EEG = pop_eegfiltnew(EEG, locutoff=1, hicutoff=40)
EEG = pop_eegfiltnew(EEG, locutoff=45, hicutoff=55, revfilt=True)
```

When `filtorder` is omitted, the transition width is recomputed from the even
default order (`3.3 / order * srate`). Numeric channel selections are
EEGLAB-facing 1-based indices. Boundary events
split continuous data before filtering, and ICA activations are cleared after
filtering. `usefftfilt=True` filters in the frequency domain with overlap-add.
A NaN affects the FFT block that contains it.
