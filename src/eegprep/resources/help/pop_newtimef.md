POP_NEWTIMEF - Plot channel or component time-frequency decomposition.

`pop_newtimef` computes an event-related spectral perturbation (ERSP) image and
inter-trial coherence (ITC) image for one channel or ICA component.

Examples:

```python
result = pop_newtimef(EEG, 1, 1, [-100, 200], [3, 0.8])
result = pop_newtimef(EEG, 0, 2, [-100, 200], [0], freqs=[4, 30], padratio=2)
```

The EEGPrep implementation provides deterministic FFT output for `cycles=[0]`
and Morlet wavelet output for non-zero `cycles`, with EEGLAB-compatible inputs,
history strings, baseline modes, bootstrap significance outputs, and image or
curve plotting. Time-warping options are not part of the standalone EEGPrep
Phase 4 implementation and fail clearly when requested.
