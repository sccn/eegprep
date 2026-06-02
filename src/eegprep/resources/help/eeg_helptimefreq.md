# EEG_HELPTIMEFREQ - Time-frequency functions

EEGPrep implements EEGLAB-style time-frequency menu wrappers for channel and
component workflows.

Implemented user-facing wrappers:

- `pop_newtimef`: compute and plot event-related spectral perturbation and
  inter-trial coherence style summaries.
- `pop_newcrossf`: compute and plot cross-channel or cross-component
  coherence, phase coherence, or cross-spectrum summaries.

These wrappers use EEGPrep's Python time-frequency backend and preserve
EEGLAB-like dialog labels and history commands where supported.

See also: POP_NEWTIMEF, POP_NEWCROSSF
