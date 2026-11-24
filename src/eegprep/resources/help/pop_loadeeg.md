# Import Neuroscan EEG data

`pop_loadeeg` reads a Neuroscan epoch `.eeg` file. Choose Auto, 16-bit, or 32-bit precision, and optionally limit the trials, event types, electrodes, or responses. Leave a box blank to import all of that subset.

Ok builds one dataset named `Neuroscan EEG data`. Each sweep gets a time-locking event. Sweeps that have a reaction time also get an RT event. Cancel closes the dialog and does not change the current dataset.
