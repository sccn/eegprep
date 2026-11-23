# pop_dipfit_settings

Configure the dataset's `EEG["dipfit"]` head-model metadata for source localization.

EEGPrep stores EEGLAB-compatible DIPFIT fields such as `hdmfile`, `mrifile`, `chanfile`, `coordformat`, `coord_transform`, and `chansel`. `standardBEM` is the default. It uses the packaged MNI boundary-element volume at fit time. A channel file whose name contains `standard-10-5-cap385` keeps that MNI model and receives the cap385-to-MNI coregistration. `standardBESA` uses the four-shell BESA leadfield. A custom head-model file is used as a boundary-element volume when it contains `bnd`, `mat`, and `cond` with type dipoli, bemcp, or openmeeg. OpenMEEG is recognized and rejected. Any other head-model file raises an error.

Changing the head-model menu fills the head-model, MRI, and channel-file fields for the selected template. A channel-file keyword transform is kept. When none matches, EEGPrep fills the montage warp from `coregister` onto the template fiducials. Co-register opens that warp dialog. The `...` button chooses channels to omit. Ok is refused while the matrix is empty unless No Co-Reg. is checked.

Creating a new BEM system matrix from an MRI is not available. Use a template model here, then fit or plot.
