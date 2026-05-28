# pop_headplot

Plots ERP or ICA maps on a static 3-D head view.

```python
figures, com = pop_headplot(EEG, typeplot=1, items=[100, 200], return_com=True)
```

`typeplot=1` plots ERP latency maps. `typeplot=0` plots component maps. EEGPrep
uses packaged Python rendering at runtime and does not depend on EEGLAB mesh
files.
