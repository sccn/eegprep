# pop_prop

Plots properties of one channel or independent component, including a scalp
location/map, ERP trace, and spectrum.

```python
fig, com = pop_prop(EEG, typecomp=1, chanorcomp=1, return_com=True)
```

Use `typecomp=1` for channels and `typecomp=0` for ICA components.

When scripting, pass `plot='off'` to build the figure without opening a window; the default `plot='on'` displays it.
