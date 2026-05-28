POP_NEWTIMEF - Plot channel or component time-frequency decomposition.

`pop_newtimef` computes an event-related spectral perturbation (ERSP) image and
inter-trial coherence (ITC) image for one channel or ICA component.

Examples:

```python
result = pop_newtimef(EEG, 1, 1, [-100, 200], [3, 0.8])
result = pop_newtimef(EEG, 0, 2, [-100, 200], [0], freqs=[4, 30], padratio=2)
```

The EEGPrep implementation provides a deterministic STFT-backed numerical core
with EEGLAB-compatible inputs, history strings, and plotting layout. EEGLAB
features that rely on permutation/bootstrap masking, time-warping, or curve
plotting currently fail clearly rather than silently changing behavior.
