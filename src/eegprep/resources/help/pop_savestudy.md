# POP_SAVESTUDY - Save an EEGPrep STUDY

`pop_savestudy` writes STUDY metadata, dataset references, and designs to a
MATLAB v6 `.study` file that EEGLAB can open. If the study exceeds the v6 size
limit, EEGPrep writes MATLAB v7.3.

## Usage

```python
STUDY, ALLEEG, com = pop_savestudy(STUDY, ALLEEG, filename="demo.study", filepath="/data")
STUDY, ALLEEG, com = pop_savestudy(STUDY, ALLEEG, savemode="resave", resavedatasets="on")
```

The function returns the saved STUDY, the dataset list, and the history command.
The returned STUDY records `filename`, `filepath`, and `saved="yes"`. It keeps
the cached channel and cluster measures that `std_rmalldatafields` removes
from the copy written to disk.

By default, only the study file is written. With `resavedatasets="on"`, EEGPrep
also writes datasets marked `saved="no"` and marks them `"yes"` in the returned
`ALLEEG`. Datasets marked `"yes"` or `"justloaded"` keep their flags.

To write JSON, use `fileformat="json"` or a `.json` filename.
`savemode="resave"` writes MATLAB at the existing study path, including when
the previous file contained JSON.

See also: POP_LOADSTUDY, POP_STUDY
