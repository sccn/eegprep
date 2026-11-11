# pop_dipfit_gridsearch

Run a coarse DIPFIT grid search for ICA components.

EEGPrep scans the requested 3-D grid, fits the best moment at each candidate location, stores `posxyz`, `momxyz`, `rv`, `diffmap`, `sourcepot`, and `datapot` in `EEG["dipfit"]["model"]`, and rejects component fits above the residual-variance threshold.

With `standardBEM`, omitted grids are `linspace(-85,85,24)` and `linspace(0,85,12)`, and candidates are kept when they fall inside the brain mesh. The leadfield is the packaged dipoli model. A spherical head model keeps the 11 by 11 by 6 grid and the single-sphere leadfield.
