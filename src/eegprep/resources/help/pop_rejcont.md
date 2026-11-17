POP_REJCONT - Reject continuous data regions by spectrum.

Usage:

    EEG, regions = pop_rejcont(EEG, "elecrange", [1, 2], "threshold", 10)
    EEG, command = pop_rejcont(EEG, return_com=True)

`pop_rejcont` scans continuous data in short windows and marks regions whose
spectral power exceeds the requested threshold. Returned regions are
EEGLAB-style 1-based sample ranges.

Use `onlyreturnselection="on"` to inspect the regions without removing data.

Frequency limits are applied to the real FFT bin frequencies. EEGLAB
`pop_rejcont.m:211` stretches those bins with `linspace` from 0 Hz to the
Nyquist frequency, so the same limits can select different bins. On
`eeglab_data.set`, 20–40 Hz at 10 dB keeps 27720 points in 8 regions;
EEGLAB keeps 28008 points in 9. The spectrum itself is `10*log10` in both.
