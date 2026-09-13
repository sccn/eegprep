# pop_chanedit

Edit channel-location metadata for the current EEG dataset.

Use `pop_chanedit(EEG, "changefield", [index, field, value])` to edit a
single channel using EEGLAB-facing 1-based channel indices. The EEGPrep port
also supports basic channel insertion, deletion, channel-location file
load/save, and coordinate conversion between Cartesian, spherical, and
topographic fields. The `shrink` option records EEGLAB's display-only
topographic shrink factor on the first channel-location record.

Use `pop_chanedit(EEG, "lookup", filename)` to preserve the dataset's channel
order while filling coordinates from matching, case-insensitive labels in a
channel-location template. EEGPrep records the template filename and nose
direction in `EEG["chaninfo"]`; labels absent from the template remain in the
dataset without invented coordinates.

The GUI presents the first channel in an EEGLAB-style channel editor. Rich
channel-table navigation is intentionally limited in this phase; command-line
calls cover the implemented edit operations.
