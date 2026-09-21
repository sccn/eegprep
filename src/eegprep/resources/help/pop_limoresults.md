# POP_LIMORESULTS - Compute LIMO group results

`pop_limoresults(STUDY, source, analysis=...)` computes a standalone
second-level result and stores it in `STUDY.limo.results`. Sources may be
numeric subject arrays, in-memory first-level models, or `.npz` files written
by `pop_limo`.

Supported analyses include contrasts, one-sample, paired and Welch two-sample
t tests, regression, one-way ANOVA, ANCOVA, repeated-measures ANOVA, and mean
or inverse-variance-weighted summaries. Bootstrap, TFCE, plotting, and direct
MATLAB LIMO `.mat` loading are not silently emulated.

See also: POP_LIMO, EEG_HELPSTUDY
