POP_REJCHAN - Reject bad channels by summary statistics.

Usage:

    EEG, indices, measure = pop_rejchan(EEG, "measure", "kurt", "threshold", 5)
    EEG, command = pop_rejchan(EEG, return_com=True)

Scripted calls use EEGLAB's default absolute threshold of `400` when
`threshold` is omitted. The interactive dialog supplies its own normalized
default threshold of `5`.

Supported measures are probability, kurtosis, spectrum, and standard deviation.
Use `indexonly="on"` to return candidate channel indices without removing them.

Channel indices in commands and dialogs are EEGLAB-facing 1-based indices.
