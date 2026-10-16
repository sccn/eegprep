# POP_CLUST - Cluster STUDY ICA components

`pop_clust` clusters the rows in `STUDY["etc"]["preclust"]["preclustdata"]`
and writes EEGLAB-style child entries into `STUDY["cluster"]`.

`Cls 1` contains the components assigned to the first centroid of the winning
start. Later cluster numbers keep that centroid order. The same `random_state`
reproduces the numbering.

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
