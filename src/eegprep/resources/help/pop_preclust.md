# POP_PRECLUST - Build component preclustering arrays

`pop_preclust` prepares ICA component features for STUDY clustering.

```python
STUDY, ALLEEG, com = pop_preclust(
    STUDY,
    ALLEEG,
    cluster_ind=1,
    preproc=[{"measure": "scalp", "npca": 3, "norm": 1, "weight": 1}],
    return_com=True,
)
```

Scalp-map features are read from loaded ICA maps. ERP, spectrum, ERSP, and ITC
features must already be present under
`STUDY["etc"]["eegprep"]["component_measures"]`, the Phase 5b component-measure
contract documented in `.notes/implementation-notes.html`.

See also: STD_PRECLUST, POP_CLUST, POP_CLUSTEDIT
