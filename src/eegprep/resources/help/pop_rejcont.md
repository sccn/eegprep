POP_REJCONT - Reject continuous data regions by spectrum.

Usage:

    EEG, regions = pop_rejcont(EEG, "elecrange", [1, 2], "threshold", 10)
    EEG, command = pop_rejcont(EEG, return_com=True)

`pop_rejcont` scans continuous data in short windows and marks regions whose
spectral power exceeds the requested threshold. Returned regions are
EEGLAB-style 1-based sample ranges.

Use `onlyreturnselection="on"` to inspect the regions without removing data.
