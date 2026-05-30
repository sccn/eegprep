# POP_IMPORTDATA - Import numeric data into an EEG dataset

`pop_importdata` creates an EEG dictionary from an array or supported numeric
file.

Usage:

```python
EEG = pop_importdata("data", "signals.tsv", "srate", 250)
EEG, com = pop_importdata("data", array, "srate", 500, return_com=True)
```

Supported array sources include Python arrays and text, CSV, TSV, NumPy, and
MATLAB array files. Provide `nbchan`, `pnts`, `srate`, `xmin`, and metadata
options when they cannot be inferred from the file.

See also: POP_FILEIO, POP_LOADSET
