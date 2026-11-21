# pop_multifit

Fit one dipole or a symmetric dipole pair for each selected ICA component.

EEGPrep combines an inside-brain grid search, nonlinear refinement, residual variance rejection, and optional dipole plotting. Results are stored in `EEG["dipfit"]["model"]`. `standardBEM` uses the packaged MNI dipoli model. `standardBESA` uses the four-shell BESA model. Both refine positions with FieldTrip's scaled BFGS procedure and solve moments by least squares.

`threshold` is a percentage, so `threshold=1` means 1% residual variance. With spherical coordinates, `rmout="on"` also removes fits beyond the 85 mm head radius. MNI fits retain DIPFIT's unconstrained position behavior.

Bilateral fits mirror the x coordinate for MNI or the y coordinate for BESA. The plot options accept MATLAB pairs such as `'normlen' 'on'`. When plotting is checked, the dipole figure is shown.
