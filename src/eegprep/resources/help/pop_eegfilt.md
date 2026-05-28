# pop_eegfilt

Legacy EEGLAB-style FIR filtering interface.

`pop_eegfilt` is kept for menu and command-history compatibility. New EEGPrep
workflows should prefer `pop_eegfiltnew` or the firfilt plugin wrappers.
The legacy `firls` and `fir1` FIR design modes are supported. The old EEGLAB
`usefft` fallback is not implemented; use `pop_eegfiltnew(..., usefftfilt=True)`
for frequency-domain FIR filtering.

```python
EEG = pop_eegfilt(EEG, 1, 40)
EEG = pop_eegfilt(EEG, 45, 55, revfilt=True)
EEG = pop_eegfilt(EEG, 1, 40, causal=True)
```

If `filtorder` is omitted, EEGPrep follows EEGLAB's legacy heuristic: use three
cycles at the lower cutoff for high-pass/band-pass filters, or three cycles at
the upper cutoff for low-pass filters, with a minimum order of 15. Set `causal`
to use one-pass causal FIR filtering; the default is zero-phase filtering.

Boundary events split continuous data before filtering, and ICA activations are
cleared after filtering.
