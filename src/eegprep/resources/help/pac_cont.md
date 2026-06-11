# PAC_CONT - Continuous phase-amplitude coupling

`pac_cont` computes sliding-window PAC for continuous vectors.

It filters the phase and amplitude signals, extracts Hilbert phase/amplitude
envelopes, and returns PAC values at output window centers. Supported methods
include modulation, PLV, correlation, and a linear-regression coefficient path.

Use `nofig="on"` for compute-only calls.

See also: PAC, STD_PAC
