# pop_chanplot

Plots precomputed STUDY channel or component measures.

```python
STUDY, com, fig = pop_chanplot(STUDY, ALLEEG, channels=[1], return_com=True)
STUDY, com, fig = pop_chanplot(
    STUDY,
    ALLEEG,
    components=[1],
    measure="erp",
    mode="components",
    return_com=True,
)
```

`pop_chanplot` reads cached measures from `STUDY.changrp` for channels and the
parent `STUDY.cluster` entry for components. Run `pop_precomp` first for
spectra, ERSP, ITC, and component measures. ERP channel plots can still fall
back to loaded epoched `ALLEEG` datasets when no cache is present.

Supported measures are ERP, spectrum, ERSP, and ITC. Component plotting is
parent-cluster based until Phase 5c adds preclustering, clustering, and cluster
editing.
