# pop_spectopo

Plots channel or component power spectra and optional scalp maps at selected
frequencies.

```python
result, com = pop_spectopo(EEG, dataflag=1, freqs=[6, 10, 22], return_com=True)
```

`dataflag=1` plots channel spectra. `dataflag=0` plots component spectra and
requires ICA activations or ICA weights.
