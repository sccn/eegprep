# pop_loadbci

Import a BCI2000-style ASCII or MATLAB data file into an EEG structure.

`pop_loadbci(filename)` is a compact standalone importer for the documented
BCI-style workflows that EEGPrep can verify without external MATLAB toolboxes.
ASCII files are read through `pop_importdata` with channel labels inferred from
the header row when present. MATLAB files are passed through the same
dictionary-oriented import path.

BCI2000 state columns in ASCII files are retained as data channels in this
scoped importer; they are not converted into EEG events.

Example:

```python
EEG, com = pop_loadbci("subject01_bci.txt", return_com=True)
```
