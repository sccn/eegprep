# pop_dipfit_gridsearch

Run a coarse DIPFIT grid search for ICA components.

EEGPrep scans the requested 3-D grid with a deterministic standalone spherical leadfield, fits the best moment at each candidate location, stores `posxyz`, `momxyz`, `rv`, `diffmap`, `sourcepot`, and `datapot` in `EEG["dipfit"]["model"]`, and rejects component fits above the residual-variance threshold.

The standalone backend is intended for practical spherical DIPFIT workflows. MRI-derived BEM/FieldTrip fitting is not silently emulated.
