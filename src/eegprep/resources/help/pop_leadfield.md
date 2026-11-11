# pop_leadfield

Compute a source-model leadfield matrix for DIPFIT/ROI workflows.

EEGPrep computes a leadfield for explicit source points supplied as an `Nx3` array, a dictionary with `{"pos": ...}`, or a simple `.npy`, `.npz`, or `.mat` file containing source positions. `standardBESA` uses the four-shell BESA model. MNI and BEM settings use an infinite-medium kernel. The result is stored in `EEG["dipfit"]["sourcemodel"]`.

FieldTrip source-model preparation, AFNI atlas clipping, and BEM head-model leadfields require explicit backend/assets and fail clearly instead of producing placeholder matrices.
