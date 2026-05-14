POP_RESAMPLE - Change the sampling rate of an EEG dataset.

Usage:
  EEG = pop_resample(EEG, freq)

Inputs:
  EEG  - EEGPrep/EEGLAB-style dataset.
  freq - new sampling rate in Hz.

Calling pop_resample(EEG) opens the EEGPrep GUI dialog. Event and urevent
latencies are adjusted to the new sampling rate.
