# pop_dipplot

Plot existing DIPFIT dipole model positions.

EEGPrep provides a standalone matplotlib view for dipoles already stored in `EEG["dipfit"]["model"]`. It plots positions, moments, residual-variance labels, optional RV filtering, projection lines, outward/normalized moment display, and packaged standard MNI MRI slices for summary/projection views.

Custom MRI volumes and FieldTrip mesh rendering are explicit asset/backend limits.
