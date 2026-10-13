# pop_erpimage

Plots an ERP image of one channel or component: single trials stacked as rows,
sorted by an event value and smoothed across trials, with the ERP underneath.

```python
result, com = pop_erpimage(EEG, 1, 14, return_com=True)
```

The arguments follow EEGLAB's order:

```python
pop_erpimage(EEG, typeplot, channel, projchan, titleplot, smooth, decimate,
             sortingtype, sortingwin, sortingeventfield, *erpimage_options)
```

- `typeplot`: 1 for channels, 0 for ICA components. The dataset must be epoched.
- `channel`: 1-based channel number(s) or labels (`"Cz"`), or component
  number(s). Several channels are averaged.
- `projchan`: channel(s) to back-project the component(s) onto, in microvolts.
  Several components need a projection channel and are summed; several
  projection channels are averaged.
- `titleplot`: figure title; by default the channel label, `Comp. 6` or
  `Comp. 6 -> Chan. 27`.
- `smooth` (default 5) and `decimate` (default 0): the moving-average width in
  trials and the window advance, as in `erpimage`.
- `sortingtype`, `sortingwin`, `sortingeventfield`: sort the trials by the
  value of an event field (for example `"latency"` of `"rt"` events) within a
  time window in ms. Text values sort alphabetically. Epochs without a
  matching event are left out, and the count is logged. Without a field the
  trials keep their order.

Any `erpimage` option can follow, as EEGLAB tokens or keywords, for example
`'phasesort', [0, 25, 10]`, `coher=[10, 10, 0.01]`, `align=float('inf')`,
`renorm='yes'`, `nosort=True` (keep the trial order but plot the event values)
or `noplot=True` (sort without drawing the values).

Unlike a bare `erpimage` call, `pop_erpimage` draws the ERP trace, the colorbar
and a scalp map (the channel location, or the component map) by default; use
`erp=False`, `cbar=False` or `plotmap=False` to leave them out. The function
returns the `erpimage` result (`result.outdata`, `result.sortidx`,
`result.figure`, ...).

```python
result, com = pop_erpimage(
    EEG, 1, "POz", None, "", 10, 1, ["rt"], [0, 1000], "latency",
    "phasesort", [0, 25, 10], return_com=True,
)
```

## Dialog

`Plot > Channel ERP image` and `Plot > Component ERP image` open EEGLAB's
dialog. The `Epoch-sorting field` and `Event type(s)` buttons list the fields
and types of the dataset's events. The phase-sorting and coherence frequencies
mirror each other when both are filled. The time limits are passed only when
you change them. `More options` takes `erpimage` options as `'key', value`
pairs. For components, the `...` button next to `Project to channel #` picks
the channel (EEGLAB disables it).

The command recorded in history (`LASTCOM`) is a Python call that you can edit
and run in `eegprep-console`. Making an ERP image does not change the dataset.

When scripting, pass `plot='off'` to build the figure without opening a
window, or `target=` a Matplotlib Figure or SubFigure to draw into it.
