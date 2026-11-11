# pop_dipfit_gridsearch

Run a coarse DIPFIT grid search for ICA components.

EEGPrep scans the requested 3-D grid with the four-shell BESA leadfield when the head model is `standardBESA`, and with an infinite-medium leadfield for MNI or BEM settings. It fits the best moment at each candidate inside the brain shell, stores `posxyz`, `momxyz`, `rv`, `diffmap`, `sourcepot`, and `datapot` in `EEG["dipfit"]["model"]`, and rejects component fits above the residual-variance threshold. The default grid is `linspace(-85,85,24)` in x and y and `linspace(0,85,12)` in z.

The standalone backend is intended for practical spherical DIPFIT workflows. MRI-derived BEM/FieldTrip fitting is not silently emulated.
