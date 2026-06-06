# pop_topochansel

`pop_topochansel(chanlocs, select)` is a compatibility channel-selection helper.
For noninteractive calls, `select` may contain 1-based channel indices or labels
and EEGPrep returns the selected indices, label list, and space-separated label
string.

The original MATLAB topographic lasso selection window is not recreated; GUI
calls use EEGPrep's existing channel selector.
