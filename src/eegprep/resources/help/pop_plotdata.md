# pop_plotdata

Plots ICA component ERP activations in a rectangular array.

```python
fig, com = pop_plotdata(EEG, components=[1, 2, 3], return_com=True)
```

This requires ICA activations or ICA weights.

When scripting, pass `plot='off'` to build the figure without opening a window; the default `plot='on'` displays it.
