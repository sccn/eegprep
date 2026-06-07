# EEG_HELPSTATISTICS - Statistical functions

EEGPrep's current statistics-facing menu surfaces focus on deterministic EEG
summary statistics exposed through plotting wrappers.

Implemented user-facing wrappers:

- `pop_signalstat`: compute channel or component signal summary statistics.
- `pop_eventstat`: summarize event counts and timing for the current dataset.
- `statcond`, `fdr`, and related statistics helpers: deterministic in-package
  condition tests and multiple-comparison utilities.
- `std_limodesign`: LIMO-compatible design matrix construction for STUDY
  factors and trial metadata.

External LIMO model fitting/result browsing and FieldTrip cluster-statistics
execution remain explicit optional-backend boundaries. EEGPrep does not create
placeholder LIMO results.

See also: POP_SIGNALSTAT, POP_EVENTSTAT, STD_LIMODESIGN
