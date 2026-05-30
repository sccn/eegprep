# pop_dipfit_settings

Configure the dataset's `EEG["dipfit"]` head-model metadata for source localization.

EEGPrep stores EEGLAB-compatible DIPFIT fields such as `hdmfile`, `mrifile`, `chanfile`, `coordformat`, `coord_transform`, and `chansel`. The standard BEM and BESA entries are symbolic template metadata in standalone EEGPrep; they do not require an EEGLAB checkout at runtime.

FieldTrip co-registration and model-file loading are not ported yet. Use this menu to prepare the dataset metadata before workflows that can consume existing DIPFIT model information.
