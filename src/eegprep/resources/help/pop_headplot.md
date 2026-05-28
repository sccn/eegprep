# pop_headplot

Plots ERP or ICA component maps on an EEGLAB-style spline-interpolated 3-D head
mesh.

```python
figures, com = pop_headplot(
    EEG,
    typeplot=1,
    items=[100, 200],
    setup={
        "splinefile": "my_montage.spl",
        "transform": [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100],
    },
    return_com=True,
)
```

`typeplot=1` plots ERP latency maps. `typeplot=0` plots component maps.
Negative component indices invert polarity. A spline setup file is required, as
in EEGLAB:

- Use `load="my_montage.spl"` to reuse an existing spline file.
- Use `setup={...}` to create a new spline file from channel locations, a head
  mesh, and a Talairach transformation matrix.
- Replaying a history command with `setup={...}` reuses an existing `.spl` file;
  pass `recompute=True` inside `setup` to force regeneration.
- The GUI exposes the same load-or-recompute workflow used by EEGLAB, including
  Manual coregistration for editing the Talairach transform against the selected
  head mesh and reference electrode file.

EEGPrep packages the standard EEGLAB head meshes needed for this workflow, so it
does not depend on an EEGLAB checkout at runtime.

The Manual coreg. window shows user electrodes in green and reference electrodes
in brown. `Align montages` fits a shared-scale transform to common labels;
`Warp montage` fits the EEGLAB-style 9-parameter transform. The transform field
is then written back to the `pop_headplot` setup dialog.
