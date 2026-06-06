TF_CYCLE_CALC - Calculate Morlet wavelet cycles from width units.

`tf_cycle_calc` converts temporal or spectral wavelet widths to Morlet cycle
counts for use with `newtimef` and `pop_newtimef`.

Examples:

```python
result = tf_cycle_calc(freqs=[8, 12, 16], width=[0.2, 0.3], width_unit="fwhm_t")
cycles = result.cycles
widths = result.widths_table
```

Supported width units are `fwhm_t`, `fwhm_f`, `2_sigma_t`, `2_sigma_f`,
`sigma_t`, `sigma_f`, and `cycles`. When two width values are supplied for more
than two frequencies, EEGPrep interpolates widths linearly or logarithmically
before calculating cycles.
