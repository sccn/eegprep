POP_ICLABEL - Classify independent components with ICLabel.

Usage:
  EEG = pop_iclabel(EEG, 'default')

Inputs:
  EEG       - EEGPrep/EEGLAB-style dataset with ICA decomposition.
  icversion - 'default', 'lite', or 'beta'.

Calling pop_iclabel(EEG) opens the EEGPrep GUI dialog. Results are stored under
EEG.etc.ic_classification.ICLabel.
