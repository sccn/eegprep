# Bolt's Performance Journal

## 2025-02-15 - Vectorized jointprob loop reduction
**Learning:** Reshaping the probability array output from `_realproba` into a 2D `(trials, points)` array allows us to vectorize the inner loop reduction using `np.sum(..., axis=1)`, resulting in a 23.3% performance improvement on large EEG datasets.
**Action:** Always check if a trial-wise calculation or aggregation within a loop can be flattened/reshaped into a multi-dimensional array to perform vectorized operations along specific axes.
