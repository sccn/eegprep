# pop_dipfit_loreta

Localize ICA components with eLORETA.

EEGPrep checks channel locations, an ICA decomposition, dipole settings, a stored leadfield, and an MNI head model. `pop_dipfit_loreta` then stops because FieldTrip source analysis and source plotting are not bundled.

Run `pop_leadfield` first when `EEG.dipfit.sourcemodel` is missing. The LORETA-KEY choice stores the in-head grid for the selected head model. Use `pop_multifit` for an equivalent dipole or a symmetric pair. Select `standardBEM` in `pop_dipfit_settings` when the head model is not already MNI.
