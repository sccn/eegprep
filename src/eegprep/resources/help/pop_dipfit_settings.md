# pop_dipfit_settings

Configure the dataset's `EEG["dipfit"]` head-model metadata for source localization.

EEGPrep stores EEGLAB-compatible DIPFIT fields such as `hdmfile`, `mrifile`, `chanfile`, `coordformat`, `coord_transform`, and `chansel`. `standardBEM` is the default. It uses the packaged MNI boundary-element volume at fit time. A channel file whose name contains `standard-10-5-cap385` keeps that MNI model and receives the cap385-to-MNI coregistration. `standardBESA` keeps the single-sphere leadfield.

Creating a new BEM system matrix from an MRI is not implemented. Interactive co-registration is not implemented. Use this menu to prepare the dataset before fitting, explicit-point leadfields, or plotting.
