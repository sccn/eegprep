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
features must already be cached on the parent `STUDY["cluster"][0]` entry,
normally by calling `pop_precomp(STUDY, ALLEEG, "components", ...)` for the
requested measures.

See also: POP_PRECOMP, STD_PRECLUST, POP_CLUST, POP_CLUSTEDIT
