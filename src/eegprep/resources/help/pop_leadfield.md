# pop_leadfield

Compute a source-model leadfield matrix for DIPFIT/ROI workflows.

EEGPrep computes a leadfield for explicit source points supplied as an `Nx3` array, a dictionary with `{"pos": ...}`, or a simple `.npy`, `.npz`, or `.mat` file containing source positions. The result is stored in `EEG["dipfit"]["sourcemodel"]`. `standardBEM` uses the packaged dipoli model. `standardBESA` uses the single-sphere leadfield.

FieldTrip MRI source-model construction and AFNI atlas clipping are not implemented.
