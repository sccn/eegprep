REREF - convert common reference EEG data to another common reference or to average reference.

Usage:

    dataout = reref(data)
    dataout, chanlocs = reref(data, refchan, 'key', 'value', ...)

Inputs:

- `data`: 2-D or 3-D EEG data matrix with channels on the first axis.
- `refchan`: reference channel number or numbers.

Reference options:

- `[]` or `None`: compute average reference.
- Single channel: re-reference to that channel.
- Multiple channels: re-reference to the average of those channels.

Optional inputs:

- `huber`: Huber average reference threshold in microvolts.
- `exclude`: channel indices excluded from re-referencing.
- `keepref`: keep reference channels in the output.
- `elocs`: current electrode-location dictionaries.
- `refloc`: previous reference-channel location to reconstruct.

Outputs:

- `dataout`: input data converted to the new reference.
- `chanlocs`: updated channel locations when location data is provided.

Average reference on float32 data is summed in float64 and stored as float32.
On a 64-channel recording with 194048 samples and a peak of 259369.28125 µV,
EEGLAB's single-precision product differs from this float32 value by at most
0.15625 µV (RMS 0.004684294629 µV). That EEGLAB result also differs by
0.109375 µV from an OpenBLAS Haswell single-precision product of the same
data (RMS 0.00235126327796 µV), and the Haswell result changes with the
thread count.
