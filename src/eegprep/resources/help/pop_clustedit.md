# POP_CLUSTEDIT - Edit and plot component clusters

`pop_clustedit` provides the API and GUI entry point for common STUDY cluster
editing actions:

- plot cluster summaries
- rename clusters
- merge clusters
- move components to an outlier cluster
- reassign components between clusters
- reject distance-based outliers

```python
STUDY, com, fig = pop_clustedit(
    STUDY,
    ALLEEG,
    action="plot",
    clusters=[2, 3],
    return_com=True,
)
```

The Phase 5c plot hook shows cluster membership summaries. Full component
measure plotting uses the Phase 5b measure outputs when that phase lands.

See also: POP_PRECLUST, POP_CLUST, STD_RENAMECLUST, STD_MERGECLUST
