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
```

Boundary events split continuous data before filtering, and ICA activations are
cleared after filtering.
