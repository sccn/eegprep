EEG_ICALABELSTAT - Summarize ICLabel component probabilities.

Usage:

    stats = eeg_icalabelstat(EEG)
    stats = eeg_icalabelstat(EEG, threshold=0.8, verbose=False)

Inputs:

- `EEG`: EEGPrep/EEGLAB-style dataset with ICLabel classifications in
  `EEG.etc.ic_classification.ICLabel`.
- `threshold`: scalar probability threshold, or one threshold per ICLabel
  class. The default is `0.9`.

Behavior:

- Prints EEGLAB-style class counts such as the number of Muscle components
  above the selected probability threshold.
- Mirrors the ICLabel plugin `eeg_icalabelstat.m` console summary, including
  the historical `IClabel` label capitalization used by EEGLAB.
- Returns a dictionary with class names, thresholds, per-class counts,
  1-based component indices above threshold, mean probabilities, dominant
  class counts, and rejected/kept tallies based on `EEG.reject.gcompreject`.
- Uses the standard ICLabel class order when the classification matrix has
  seven columns and class names are not stored.

Example:

    EEG = pop_iclabel(EEG, 'default')
    stats = eeg_icalabelstat(EEG, 0.9)
    print(stats["counts"])
