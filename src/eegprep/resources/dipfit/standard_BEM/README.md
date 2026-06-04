# Standard MNI MRI Resource

`standard_mri_mni.npz` is an EEGPrep-packaged conversion of the MNI standard
MRI volume distributed by SCCN's DIPFIT plugin as
`standard_BEM/standard_mri.mat`.

The NPZ contains the fields needed for native Python dashboard rendering:

- `anatomy`: 3-D uint8 MRI anatomy volume
- `transform`: voxel-to-MNI homogeneous transform
- `xgrid`, `ygrid`, `zgrid`: 1-based MRI grid coordinates from the source file

It is used only for visualization of localized dipoles in EEGLAB-compatible
viewprops dashboards. It is not a volume-conductor model and is not used for
FieldTrip/DIPFIT fitting.

Provenance: converted from `sccn/dipfit` commit `b0b660e`.
The accompanying `standard_mri_license.txt` preserves the MNI permission notice
used for EEGPrep's packaged MNI resources.
