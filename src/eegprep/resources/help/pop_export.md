# POP_EXPORT - Export EEG data to text

`pop_export` writes EEG data or ICA activity to a text file.

Usage:

```python
com = pop_export(EEG, "data.tsv", "transpose", "on")
com = pop_export(EEG, "data.tsv", "expr", "x = x * 1e6")
```

Supported options include `ica`, `time`, `timeunit`, `elec`, `transpose`,
`erp`, `precision`, `separator`, and `expr`.

`expr` applies a numeric Python expression to the exported array `x` before the
optional time row is added. EEGPrep supports arithmetic, indexing, comparisons,
and selected NumPy numeric functions (`abs`, `clip`, `exp`, `log`, `log10`,
`nan_to_num`, `sqrt`, with or without the `np.` prefix). Expressions may either
return the transformed array or assign it to `x`, for example `x * 1e6` or
`x = np.log10(abs(x) + 1)`. Function calls are sandboxed: `clip` accepts
`a_min`/`a_max` and `min`/`max` keyword arguments, and `nan_to_num` accepts
`nan`/`posinf`/`neginf`. Side-effect keywords such as `out` are not available.
For example, `x = np.clip(x, a_min=0, a_max=100)` is supported. Power operators
require small numeric-constant exponents such as `x ** 2`; dynamic or very
large exponents are rejected before evaluation. The expression must leave a 2-D
or 3-D array.

The main-window export action prompts for an output file and records the
resulting command in session history. Use BIDS export for dataset sidecars and
structured folder output.

See also: POP_EXPORTBIDS, POP_EXPEVENTS, POP_WRITEEEG
