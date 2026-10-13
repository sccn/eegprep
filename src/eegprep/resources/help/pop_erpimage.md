# pop_erpimage

Plots an ERP image for one channel or component.

```python
result, com = pop_erpimage(EEG, typeplot=1, index=1, return_com=True)
```

`typeplot=1` selects channel data. `typeplot=0` selects ICA component data. The
dataset must be epoched.

Supported plot options include `title`, `limits`, `caxis`, `cbar`, `erp`,
`vert`, `smooth`, `decimate`, `sort_values`, and component projection through
`projchan`. Channel plots draw a small scalp map above the image with the
plotted electrode marked; pass `plotmap=False` to suppress it.

By default the color axis is symmetric about zero (as in EEGLAB); pass `caxis`
to set explicit limits. A solid vertical line marks time zero on both the image
and the ERP trace.

Event-field sorting is available with EEGLAB-style names:

```python
result, com = pop_erpimage(
    EEG,
    typeplot=1,
    index=1,
    sortingeventfield="rt",
    sortingtype=["target"],
    sortingwin=[0, 800],
    return_com=True,
)
```

Use `nosort=True` to keep trials in dataset order. `noplot` is accepted and
kept in the history command, but it does not change the plot yet.
`renorm="yes"` rescales finite event values to 0..1.

The image itself comes from `erpimage`, which follows EEGLAB's sorting,
moving-average smoothing and decimation: `smooth` is the moving-window width in
trials, `decimate` is the window advance, and edge windows that would be
incomplete are trimmed. Trials whose sorting value is missing are left out.

This dialog does not expose EEGLAB's phase sorting, coherence, spectrum inset,
amplitude image, event alignment or renormalization formulas. Those options
raise a `ValueError` that points to `erpimage`, which supports all of them:

```python
from eegprep import erpimage

erpimage(data, rt, times, "", 5, 1, "phasesort", [400, 0, 10], "coher", [10, 10, 0.01])
```

When scripting, pass `plot='off'` to build the figure without opening a window; the default `plot='on'` displays it.
