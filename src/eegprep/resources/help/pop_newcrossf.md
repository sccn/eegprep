POP_NEWCROSSF - Plot channel or component cross-coherence.

`pop_newcrossf` computes time-frequency coherence between two channels or two
ICA components and plots coherence amplitude and phase.

Examples:

```python
result = pop_newcrossf(EEG, 1, 1, 2, [-100, 200], [3, 0.5])
result = pop_newcrossf(EEG, 0, 1, 2, [-100, 200], [0], type="coher")
```

Supported deterministic coherence modes are `phasecoher`, `coher`, and
`crossspec`. Single-trial continuous inputs use EEGLAB's cross-spectrum mode.
EEGLAB bootstrap and shuffle significance options are not yet ported and raise
clear `NotImplementedError` messages.
