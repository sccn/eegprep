# pop_erpimage

Plots an ERP image for one channel or component.

```python
result, com = pop_erpimage(EEG, typeplot=1, index=1, return_com=True)
```

`typeplot=1` selects channel data. `typeplot=0` selects ICA component data. The
dataset must be epoched.
