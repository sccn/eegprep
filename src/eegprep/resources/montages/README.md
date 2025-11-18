These are layout files for specific known EEG montages / cap designs;
these are used to infer channel locations when only labels are known
but match one of these montage files well.

These files are regular MATLAB files and can be loaded in
MATLAB via the command:
`locs = load('/path/to/resources/montages/my-file.locs', '-mat')`

You can create your own by following the structure of existing files
closely and entering your own labels, coordinates, etc.
