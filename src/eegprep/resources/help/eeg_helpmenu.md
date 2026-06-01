# EEG_HELPMENU - EEGPrep menu overview

The EEGPrep main window mirrors the EEGLAB menu layout while exposing only
standalone EEGPrep workflows at runtime.

Top-level menus:

- File: import, export, dataset save/load, STUDY save/load, history scripts,
  preferences, extension inventory, and quit actions.
- Edit: dataset metadata, event fields, event values, comments, channel
  locations, selection, copy, append, and delete actions.
- Tools: resampling, filtering, re-referencing, interpolation, rejection, ICA,
  component removal, epoching, and baseline removal workflows.
- Plot: channel, component, ERP, spectra, time-frequency, statistics, DIPFIT,
  ICLabel, and viewprops plotting surfaces implemented in this repository.
- Study: STUDY/group-level workflows for design, measure precompute, plotting,
  clustering, and study metadata.
- Datasets: current dataset retrieval and multi-dataset selection.
- Help: packaged EEGPrep help resources, docs links, update links, issue
  reporting, and contact actions.

Menu placeholders are machine-readable and carry a target phase or an explicit
exclusion reason for workflows that cannot be packaged in EEGPrep.

See also: EEGPREP, EEG_HELPHELP
