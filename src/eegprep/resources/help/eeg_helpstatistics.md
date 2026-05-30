# EEG_HELPSTATISTICS - Statistical functions

EEGPrep's current statistics-facing menu surfaces focus on deterministic EEG
summary statistics exposed through plotting wrappers.

Implemented user-facing wrappers:

- `pop_signalstat`: compute channel or component signal summary statistics.
- `pop_eventstat`: summarize event counts and timing for the current dataset.

Full EEGLAB STUDY statistics and FieldTrip/LIMO-style statistical workflows are
outside Phase 6a and should remain documented with their owning implementation
phase when they are added.

See also: POP_SIGNALSTAT, POP_EVENTSTAT
