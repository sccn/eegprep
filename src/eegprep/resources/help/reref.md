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
