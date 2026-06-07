# pop_dipfit_settings

Configure the dataset's `EEG["dipfit"]` head-model metadata for source localization.

EEGPrep stores EEGLAB-compatible DIPFIT fields such as `hdmfile`, `mrifile`, `chanfile`, `coordformat`, `coord_transform`, and `chansel`. The standard BESA path can be used by EEGPrep's standalone spherical fitting backend; the standard BEM fields identify the familiar EEGLAB template metadata without requiring an EEGLAB checkout at runtime.

MRI/BEM co-registration and model-file loading remain explicit backend limits. Use this menu to prepare the dataset metadata before spherical fitting, explicit-point leadfields, or plotting workflows.
