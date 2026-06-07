# POP_LOADSET_H5 - Load an HDF5-backed EEGLAB dataset

`pop_loadset_h5` loads HDF5-backed EEGLAB `.set` files and returns an EEGPrep
EEG dictionary.

Usage:

```python
EEG = pop_loadset_h5("sample_data/eeglab_data_hdf5.set")
```

Most users should call `pop_loadset`. It automatically falls back to
`pop_loadset_h5` when a `.set` file uses the HDF5 layout. Use this helper
directly only when you specifically want to exercise the HDF5 loader path.

The loader normalizes ICA channel indices and then runs EEGPrep's dataset
checks before returning the dataset.

See also: POP_LOADSET, POP_SAVESET
