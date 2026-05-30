# POP_EXPICA - Export ICA weights

`pop_expica` exports ICA weight matrices from the current dataset.

Usage:

```python
com = pop_expica(EEG, "ica_weights.txt", "weights")
com = pop_expica(EEG, "ica_inverse.txt", "inv")
```

The File > Export menu uses the `weights` variant for the ICA weight matrix and
the `inv` variant for the inverse weight matrix.

See also: POP_RUNICA, POP_SUBCOMP
