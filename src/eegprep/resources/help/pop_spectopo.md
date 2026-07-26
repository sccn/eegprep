# pop_spectopo

Plots channel or component power spectra and optional scalp maps at selected
frequencies.

```python
result, com = pop_spectopo(EEG, dataflag=1, freqs=[6, 10, 22], return_com=True)
```

`dataflag=1` plots channel spectra. `dataflag=0` plots component spectra and
requires ICA activations or ICA weights.

For `dataflag=0`, the spectra panel overlays the bold black RMS-power curve of
the channel data, marks the analysis frequency with a vertical line, and draws
the scalp maps of the `nicamaps` components with the most power at that
frequency (labeled by component index) alongside the composite
power-at-frequency map, each joined to the marker by a leader line — matching
EEGLAB `spectopo`. Set `icamaps` to map specific components instead.

The GUI's "Spectral and scalp map options" field accepts either Python
`key=value` pairs (e.g. `electrodes='off', style='blank'`) or the classic
`'key', value` pairs (e.g. `'electrodes', 'off', 'style', 'blank'`). Multiple
entries in either style are separated by commas.
