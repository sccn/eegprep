# pop_dipfit_gridsearch

Run a coarse DIPFIT grid search for ICA components.

EEGPrep scans the requested 3-D grid, fits the best moment at each candidate inside the head model, stores `posxyz`, `momxyz`, `rv`, `diffmap`, `sourcepot`, and `datapot` in `EEG["dipfit"]["model"]`, and rejects component fits above the residual-variance threshold. The default grid is `linspace(-85,85,24)` in x and y and `linspace(0,85,12)` in z.

With `standardBEM`, candidates are kept when they fall inside the brain mesh and the leadfield is the packaged dipoli model. `standardBESA` uses the four-shell BESA leadfield and keeps candidates inside the brain shell.
