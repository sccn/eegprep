# pop_comperp

Computes and plots grand-average ERPs across loaded datasets.

```python
result, com = pop_comperp(ALLEEG, flag=1, datadd=[1, 2], return_com=True)
```

`flag=1` uses channels. `flag=0` uses ICA components. Dataset indices are
EEGLAB-facing and 1-based.

The Phase 4 implementation wires dataset selection, channel/component subset,
RMS mode, low-pass display filtering, and plottopo options. Statistical
highlighting and the full EEGLAB checkbox matrix for all ERP/std display
variants are intentionally deferred; selecting those unsupported options from
the GUI raises a clear `NotImplementedError`.
