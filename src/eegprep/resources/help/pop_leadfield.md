# pop_leadfield

Compute a source-model leadfield matrix for DIPFIT/ROI workflows.

EEGPrep computes a leadfield for explicit source points supplied as an `Nx3` array, a dictionary with `{"pos": ...}`, the name `loreta-key`, or a simple `.npy`, `.npz`, or `.mat` file containing source positions. The result is stored in `EEG["dipfit"]["sourcemodel"]`. `standardBEM` uses the packaged dipoli model. `standardBESA` uses the four-shell BESA model.

`loreta-key` builds a 20 mm grid from -80 mm to 80 mm and keeps points inside a 75 mm radius. The leadfield uses the selected head model. FieldTrip's LORETA-KEY atlas is not bundled. Colin27, ICBM152, and AFNI choices need a source-model file.

FieldTrip MRI source-model construction and AFNI atlas clipping are not implemented.
