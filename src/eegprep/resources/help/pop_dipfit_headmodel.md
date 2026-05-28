# pop_dipfit_headmodel

Create a subject-specific DIPFIT head model from an anatomical MRI.

In EEGLAB this workflow delegates MRI reading, segmentation, surface extraction, and head-model preparation to FieldTrip. EEGPrep currently collects the same user-facing parameters but raises a clear missing-backend message instead of silently doing nothing.
