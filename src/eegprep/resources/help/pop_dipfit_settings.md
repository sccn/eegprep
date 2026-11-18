# pop_dipfit_settings

Configure the dataset's `EEG["dipfit"]` head-model metadata for source localization.

Changing the head-model menu fills the head-model, MRI, and channel-file fields for the selected template, and sets the coordinate format and co-registration fields to match. Browse is enabled for a custom head model. Co-register stays disabled because the FieldTrip coregistration window is not bundled. No Co-Reg. is checked when the transform is empty.

EEGPrep stores `hdmfile`, `mrifile`, `chanfile`, `coordformat`, `coord_transform`, and `chansel`.
