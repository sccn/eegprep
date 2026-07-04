## 2025-05-15 - [Vectorizing jointprob trial loop]
**Learning:** Reshaping 1D probability arrays (from `_realproba`) to (trials, points) allows for efficient vectorized summation of logs across the trials axis, avoiding the overhead of Python loops and slice-based indexing in `jointprob`.
**Action:** Always check if flat data arrays resulting from Fortran-order reshaping (like EEGLAB's `_realproba` output) can be easily reshaped back into multi-dimensional arrays for vectorized aggregation.
