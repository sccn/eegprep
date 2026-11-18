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

Each call appends the applied design to `EEG["etc"]["eegfiltnew"]`. Another
filter appends a second record. The record includes the sampling rate, the
1-based channels that were filtered, `phase` (`zerophase` or `minphase`), the
passband edges, the -6 dB cutoffs, the order, the transition width, `revfilt`,
and the coefficients that were applied. A missing passband edge is NaN.
A highpass is stored as the reversed lowpass that was designed: `locutoff`
is NaN, `hicutoff` is the passband edge, and `revfilt` is true.
`design_eegfiltnew` is the designer used for that record. `return_com=True`
still returns `(EEG, com)`, and `com` still lists the requested options.
