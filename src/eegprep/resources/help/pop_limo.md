# POP_LIMO - Fit first-level LIMO models

`pop_limo(STUDY, ALLEEG, ...)` fits the active STUDY design to every
selected subject's epoched data. Supported methods are ordinary least squares
(`OLS`), LIMO PCOut-weighted least squares (`WLS`), and feature-wise
Tukey-bisquare iteratively reweighted least squares (`IRLS`). Use `timelim` to
select milliseconds and `outputdir` to save safe, versioned `.npz` models.

The returned models include the exact design, parameter names, betas, fitted
values, residuals, R², residual variance, standard errors, t and p values, and
robust weights. Channel and component time-domain models are supported.
Bootstrap, TFCE, non-time-domain measures, and MATLAB LIMO `.mat` interchange
remain explicit unsupported boundaries.

See also: POP_LIMORESULTS, STD_LIMODESIGN, EEG_HELPSTATISTICS
