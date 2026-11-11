# pop_dipfit_settings

Configure the dataset's `EEG["dipfit"]` head-model metadata for source localization.

The dialog opens on Template Spherical Four-Shell (BESA). That choice stores the `dipfitdefs.m` radii 71, 72, 79, and 85 mm and conductivities 0.33, 1.0, 0.0042, and 0.33. The current spherical fitter is an infinite homogeneous medium. It uses the outer 85 mm radius as the head boundary and does not apply those conductivities. This default stays until the MNI boundary-element forward model is available. Changing the head-model menu fills the file fields for the selected template.

The MNI boundary-element file `standard_vol.mat` is not bundled. Selecting that template keeps its file names, and a later fit asks you to choose the BESA sphere. Browse is enabled for a custom head model. Co-register stays disabled because the FieldTrip coregistration window is not bundled. No Co-Reg. is checked when the transform is empty.

EEGPrep stores `hdmfile`, `mrifile`, `chanfile`, `coordformat`, `coord_transform`, `chansel`, and, for the BESA template, `vol`.
