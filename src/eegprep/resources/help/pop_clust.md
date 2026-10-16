# POP_CLUST - Cluster STUDY ICA components

`pop_clust` clusters the rows in `STUDY["etc"]["preclust"]["preclustdata"]`
and writes EEGLAB-style child entries into `STUDY["cluster"]`.

`algorithm="kmeanscluster"` numbers `Cls k` from the `k`-th preclustering row.
`algorithm="kmeans"` numbers `Cls k` from centroid `k` of the lowest-SSE seeded
start. The same `random_state` reproduces the `kmeans` numbering.

```python
STUDY, com = pop_clust(
    STUDY,
    ALLEEG,
    algorithm="kmeans",
    clus_num=4,
    random_state=0,
    return_com=True,
)
```

Run `pop_preclust` first. This dialog runs `kmeans` or `kmeanscluster`.

See also: POP_PRECLUST, POP_CLUSTEDIT
