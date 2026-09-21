# pop_plotdata

Plots selected channel or ICA-component activity. Selected trials are averaged
by default, or overlaid per channel/component with `singletrials=1`.

```python
channel_fig, com = pop_plotdata(
    EEG, 1, [1, 2, 3], [2, 5, 7], "Selected channel ERPs", return_com=True
)
component_fig = pop_plotdata(EEG, components=[1, 2, 3], plot="off")
```

`typeplot=1` selects channels and `typeplot=0` selects components. The
`components=` spelling is a convenience for component mode. All indices are
1-based. Component mode requires ICA activations or ICA weights.

When scripting, pass `plot='off'` to build the figure without opening a window; the default `plot='on'` displays it.
