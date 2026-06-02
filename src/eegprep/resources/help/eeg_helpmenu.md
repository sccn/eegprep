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
- Study: Phase 5 STUDY/group-level workflows. Implemented actions are enabled;
  pending design, precompute, and clustering actions remain explicit
  placeholders until their phase lands.
- Datasets: current dataset retrieval and multi-dataset selection.
- Help: packaged EEGPrep help resources, docs links, update links, issue
  reporting, and contact actions.

Menu placeholders are machine-readable and carry a target phase or an explicit
exclusion reason. EEGBrowser/eegplot-style scrolling workflows are excluded
from the current parity scope.

See also: EEGPREP, EEG_HELPHELP
