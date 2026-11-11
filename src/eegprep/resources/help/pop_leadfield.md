# pop_leadfield

Compute a source-model leadfield matrix for DIPFIT/ROI workflows.

EEGPrep can compute a standalone spherical leadfield for explicit source points supplied as an `Nx3` array, a dictionary with `{"pos": ...}`, the name `loreta-key`, or a simple `.npy`, `.npz`, or `.mat` file containing source positions. The result is stored in `EEG["dipfit"]["sourcemodel"]`.

`loreta-key` builds a 20 mm grid from -80 mm to 80 mm and keeps points inside a 75 mm radius. FieldTrip's LORETA-KEY atlas is not bundled. Colin27, ICBM152, and AFNI choices need a source-model file.

FieldTrip source-model preparation, AFNI atlas clipping, and BEM head-model leadfields require explicit backend/assets and fail clearly instead of producing placeholder matrices.
