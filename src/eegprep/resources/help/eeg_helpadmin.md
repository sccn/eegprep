# EEG_HELPADMIN - Admin functions

Admin functions manage EEGPrep sessions, options, history, and the main
EEGLAB-style user interface.

Implemented admin surfaces include:

- `eeglab` / `gui`: launch the EEGPrep main window.
- `eegprep-console`: launch the shared GUI plus Python workspace.
- `EEGPrepSession`: synchronize `EEG`, `ALLEEG`, `CURRENTSET`, `LASTCOM`,
  `ALLCOM`, `STUDY`, and `CURRENTSTUDY`.
- `eeg_checkset`: normalize and validate EEG dictionary fields.
- `eeg_store`, `eeg_retrieve`, and `pop_delset`: manage loaded datasets.
- `pop_editoptions`: update EEGPrep menu and memory options.
- `pop_saveh` and `pop_runscript`: save and run history scripts.

Runtime admin help is packaged with EEGPrep. Development parity checks may use
the vendored EEGLAB tree, but installed admin actions do not read from it.

See also: POP_EDITOPTIONS, POP_SAVEH, POP_RUNSCRIPT
