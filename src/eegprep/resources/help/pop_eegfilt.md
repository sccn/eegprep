# pop_eegfilt

Legacy EEGLAB-style FIR filtering interface.

`pop_eegfilt` is kept for menu and command-history compatibility. New EEGPrep
workflows should prefer `pop_eegfiltnew` or the firfilt plugin wrappers.

```python
EEG = pop_eegfilt(EEG, 1, 40)
EEG = pop_eegfilt(EEG, 45, 55, revfilt=True)
```

Boundary events split continuous data before filtering, and ICA activations are
cleared after filtering.
