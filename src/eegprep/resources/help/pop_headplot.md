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
- The GUI exposes the same load-or-recompute workflow used by EEGLAB.

EEGPrep packages the standard EEGLAB head meshes needed for this workflow, so it
does not depend on an EEGLAB checkout at runtime.

Current limitation: EEGLAB's interactive manual 3-D coregistration editor is not
yet ported. EEGPrep supports transform-entry setup and shows a clear message if
the Manual coreg. button is pressed.
