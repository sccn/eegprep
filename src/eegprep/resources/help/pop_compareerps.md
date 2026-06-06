# pop_compareerps

`pop_compareerps(ALLEEG, setlist, chansubset, plottitle)` plots ERP traces for
selected epoched datasets. It is a legacy compatibility entry point backed by
EEGPrep's `pop_comperp` implementation.

Dataset indices are EEGLAB-facing 1-based values. `chansubset` is also
1-based; pass an empty list to include all channels.
