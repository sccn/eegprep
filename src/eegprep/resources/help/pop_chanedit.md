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

The GUI presents the first channel in an EEGLAB-style channel editor. Type a
channel number and press Enter, or use the arrows, to show another channel.
The number stays inside 1 through the channel count. An edit is kept on the
channel you leave, and Ok records every edit. Plotting and file import stay
unavailable in this phase; command-line calls cover the implemented edit
operations.
