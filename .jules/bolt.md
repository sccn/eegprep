## 2026-06-24 - [Vectorizing RNG calls in RANSAC/ICA]
**Learning:** Scalar calls to `stream.rand()` in tight loops (like Fisher-Yates shuffle) are significantly slower than a single vectorized call to `stream.rand(n)`. In this codebase, `rand_permutation` and `rand_sample` were bottlenecks during ICA training because they performed thousands of individual RNG calls per step.

**Action:** Vectorize random number generation in `rand_permutation` and `rand_sample` by pre-generating the required number of random values. Use `np.floor(ks * rands + 0.5)` to maintain parity with the existing `round_mat` logic for positive values.
