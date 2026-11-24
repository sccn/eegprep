# Spherical BESA MRI Resource

`avg152t1.npz` is an EEGPrep-packaged conversion of the ICBM average-152 T1
distributed by SCCN's DIPFIT plugin as `standard_BESA/avg152t1.mat`
(MINC source `avg152T1.mnc`).

Spherical dipole plots use this volume. MNI plots use
`standard_BEM/standard_mri_mni.npz` instead.

The NPZ contains:

- `anatomy`: 91 x 109 x 91 float32 volume, scaled to 0..1
- `transform`: voxel-to-world homogeneous transform (2 mm voxels)
- `xgrid`, `ygrid`, `zgrid`: 1-based grid coordinates from the source file

Provenance: converted from the `sccn/dipfit` copy of `avg152t1.mat`.
