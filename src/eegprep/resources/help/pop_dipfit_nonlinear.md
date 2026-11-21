# pop_dipfit_nonlinear

Refine the positions and moments of dipoles for one ICA component.

`standardBEM` uses the packaged MNI dipoli model. `standardBESA` uses the four-shell BESA model. Both use FieldTrip's `fminunc` procedure with 10 mm typical coordinates, scaled BFGS updates, a cubic Wolfe line search, and FieldTrip stopping tolerances. Fits are unconstrained by default, as in FieldTrip.

Use `nonlinear="no"` to estimate moments at fixed positions. Results are stored in `EEG["dipfit"]["model"]`, with replayable command history. All channel and component selections use the existing DIPFIT settings.

The manual dialog preserves EEGLAB's fields. Pressing OK applies the displayed positions and selections, then runs the standalone fit.
