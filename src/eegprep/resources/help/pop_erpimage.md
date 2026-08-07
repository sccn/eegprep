# pop_erpimage

Plots an ERP image for one channel or component.

```python
result, com = pop_erpimage(EEG, typeplot=1, index=1, return_com=True)
```

`typeplot=1` selects channel data. `typeplot=0` selects ICA component data. The
dataset must be epoched.

Supported plot options include `title`, `limits`, `caxis`, `cbar`, `erp`,
`vert`, `smooth`, `decimate`, `sort_values`, and component projection through
`projchan`.

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

Use `nosort=True` to disable value sorting and `noplot=True` to suppress the
event-value curve option from history replay. `renorm="yes"` rescales finite
event values to 0..1; custom renormalization formulas are not evaluated.

EEGPrep does not implement EEGLAB's phase-sorting, coherence, spectrum inset,
amplitude-image, or event-alignment workflows in this standalone ERP-image
wrapper. Those options raise clear `ValueError` messages instead of falling
through to MATLAB-only behavior.

When scripting, pass `plot='off'` to build the figure without opening a window; the default `plot='on'` displays it.
