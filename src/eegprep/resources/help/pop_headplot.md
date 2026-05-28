# pop_headplot

Plots ERP or ICA maps on a static 3-D head view.

```python
figures, com = pop_headplot(EEG, typeplot=1, items=[100, 200], return_com=True)
```

`typeplot=1` plots ERP latency maps. `typeplot=0` plots component maps. EEGPrep
uses packaged Python rendering at runtime and does not depend on EEGLAB mesh
files.

Note: EEGLAB's full `headplot` renders spline-interpolated scalp surfaces from
mesh/spline files. This Phase 4 EEGPrep wrapper provides a standalone static
3-D channel map and raises clearly when usable channel coordinates are missing.
