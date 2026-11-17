# pop_chanedit

Edit channel-location metadata for the current EEG dataset.

Use `pop_chanedit(EEG, "changefield", [index, field, value])` to edit a
single channel using EEGLAB-facing 1-based channel indices. The editor
also supports channel insertion, deletion, channel-location file
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
channel you leave, and Ok records every edit in order. Read locations imports a
file in channel order; Look up locs matches the existing channel labels to a
template. Labels without coordinates trigger a lookup prompt with a packaged
montage, so no EEGLAB installation is needed.

The action buttons also provide 2-D/3-D plots, head-center optimization,
rotation, coordinate conversions, head-radius scaling, channel type/reference
labels, and file export. Transform axes accepts numeric vector assignments,
such as `TMP=X; X=-Y; Y=TMP`; arbitrary MATLAB/Python code and indexed assignments
are not supported. Set reference changes metadata; use Re-reference to change
the data.

Insert/Append creates a non-data location, useful for a reference electrode.
It is stored in `EEG["chaninfo"]["nodatchans"]`. Deleting a data channel requires
`pop_select`; accepting a location count that differs from the data raises an
error. Cancel discards metadata edits. A file explicitly saved from the editor
remains on disk.

Coordinates must be finite numbers or blank. In particular, `theta='abc'` is
rejected with a field-specific error, unlike the audited EEGLAB behavior.
Polar-only files use an 85 mm head radius when no physical radius is supplied.
A template lookup with no matching labels preserves existing coordinates.
