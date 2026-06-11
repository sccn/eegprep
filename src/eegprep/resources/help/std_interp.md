# STD_INTERP - Interpolate STUDY channels

`std_interp` interpolates requested missing channels across all loaded STUDY
datasets using EEGPrep's channel interpolation backend.

When you pass channel labels, EEGPrep keeps each dataset's existing channels
and adds only the requested missing locations from the merged STUDY channel
set. Use `method="spherical"` for the default EEGLAB-like interpolation path.

See also: POP_INTERP, STD_PREPARE_NEIGHBORS
