# pop_comperp

Computes and plots grand-average ERPs across loaded datasets.

```python
result, com = pop_comperp(ALLEEG, flag=1, datadd=[1, 2], return_com=True)
```

`flag=1` uses channels. `flag=0` uses ICA components. Dataset indices are
EEGLAB-facing and 1-based.

Supported options include dataset selection, channel/component subset, `mode`
(`ave` or `rms`), `lowpass`, `title`, `tlim`, `ylim`, `alpha`, and the EEGLAB
display toggles for added, subtracted, and difference averages, standard
deviation traces, and per-dataset ERP traces.

When `alpha` is supplied, EEGPrep runs deterministic t-tests across datasets
and highlights significant time regions in the plot. A paired t-test is used
when both `datadd` and `datsub` are supplied; otherwise added datasets are
tested against zero. At least two datasets are required for significance
testing.

Unsupported option names now raise `ValueError` with the unsupported keys
listed. STUDY-level ERP statistics and EEGLAB plot callbacks that depend on
MATLAB workspace state remain outside this standalone wrapper.
