POP_RUNICA - Run ICA decomposition on an EEG dataset.

Usage:
  EEG = pop_runica(EEG, 'icatype', 'runica', ...)

EEGPrep currently supports the runica backend for this pop wrapper. Calling
pop_runica(EEG) opens a GUI dialog for selecting the runica option preset,
editing command-line options, optionally reordering components by variance, and
restricting ICA to selected channels.
