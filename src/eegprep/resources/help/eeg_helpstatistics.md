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
- `std_limo` and `pop_limo`: mass-univariate OLS, PCOut-weighted WLS, and
  Tukey-bisquare IRLS first-level models.
- `std_limoresults` and `pop_limoresults`: contrasts, t tests, regression,
  ANOVA/ANCOVA, repeated-measures ANOVA, and weighted summaries.

MATLAB LIMO `.mat` interchange, bootstrap, TFCE, LIMO's plotting interface,
and FieldTrip cluster-statistics execution remain explicit boundaries.

See also: POP_SIGNALSTAT, POP_EVENTSTAT, STD_LIMODESIGN, POP_LIMO, POP_LIMORESULTS
