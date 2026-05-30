# POP_CLUST - Cluster STUDY ICA components

`pop_clust` clusters the rows in `STUDY["etc"]["preclust"]["preclustdata"]`
and writes EEGLAB-style child entries into `STUDY["cluster"]`.

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

Run `pop_preclust` first. EEGPrep currently supports deterministic k-means and
the `kmeanscluster` alias; neural-network, affinity-propagation, and optimal
k-means branches are not ported in Phase 5c.

See also: POP_PRECLUST, POP_CLUSTEDIT
