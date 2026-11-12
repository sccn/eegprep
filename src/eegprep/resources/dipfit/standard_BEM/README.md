# Standard MNI MRI Resource

`standard_mri_mni.npz` is an EEGPrep-packaged conversion of the MNI standard
MRI volume distributed by SCCN's DIPFIT plugin as
`standard_BEM/standard_mri.mat`.

The NPZ contains the fields needed for native Python dashboard rendering:

- `anatomy`: 3-D uint8 MRI anatomy volume
- `transform`: voxel-to-MNI homogeneous transform
- `xgrid`, `ygrid`, `zgrid`: 1-based MRI grid coordinates from the source file

It is used for visualization of localized dipoles in viewprops dashboards.
It is not the volume conductor.

`standard_vol.npz` is the DIPFIT `standard_BEM/standard_vol.mat` dipoli model
distributed with the plugin (MATLAB 5 file dated 2012-03-09, three
compartments, conductivities 0.33 / 0.004125 / 0.33 S/m, and the 3000 by 3000
system matrix). EEGPrep applies that matrix to EEG electrodes. It does not
rebuild it. Triangle indices in the NPZ are the original one-based MATLAB
indices. `elec/standard_1005.elc` is the DIPFIT template electrode file for
this model (ASA, millimetres, no separate copyright header).

Provenance: MRI converted from `sccn/dipfit` commit `b0b660e`. The volume and
electrode file are the copies shipped by that plugin. The volume is derived
from the same MNI anatomy as the MRI. The accompanying `standard_mri_license.txt`
preserves the MNI permission notice used for these packaged resources.
