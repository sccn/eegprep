# pop_chanedit

Edit channel-location metadata for the current EEG dataset.

Use `pop_chanedit(EEG, "changefield", [index, field, value])` to edit a
single channel using EEGLAB-facing 1-based channel indices. The EEGPrep port
also supports basic channel insertion, deletion, channel-location file
load/save, and coordinate conversion between Cartesian, spherical, and
topographic fields.

The GUI presents the first channel in an EEGLAB-style channel editor. Rich
channel-table navigation is intentionally limited in this phase; command-line
calls cover the implemented edit operations.
