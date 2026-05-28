# pop_chanplot

Plots STUDY channel measures from loaded datasets.

```python
STUDY, com, fig = pop_chanplot(STUDY, ALLEEG, channels=[1], return_com=True)
```

Phase 4 supports ERP channel-measure plotting from epoched `ALLEEG` datasets.
Full STUDY precompute, clustering, and measure-management workflows remain in
the STUDY phase.

The GUI currently exposes channel selection and the ERP measure. Additional
STUDY measure controls such as spectra, ERSP, ITC, ERPimage, clustering, and
design-specific plotting are Phase 5 work.
