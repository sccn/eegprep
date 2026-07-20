# pop_spectopo

Plots channel or component power spectra and optional scalp maps at selected
frequencies.

```python
result, com = pop_spectopo(EEG, dataflag=1, freqs=[6, 10, 22], return_com=True)
```

`dataflag=1` plots channel spectra. `dataflag=0` plots component spectra and
requires ICA activations or ICA weights.

The GUI's "Spectral and scalp map options" field accepts either Python
`key=value` pairs (e.g. `electrodes='off', style='blank'`) or the classic
`'key', value` pairs (e.g. `'electrodes', 'off', 'style', 'blank'`). Multiple
entries in either style are separated by commas.
