
## 2025-05-14 - Vectorized Trial Loop in jointprob
**Learning:** Vectorizing the inner trial loop in `jointprob` by reshaping the flattened probabilities and using `np.sum(..., axis=1)` provides a significant speedup (~30%) for typical trial counts (e.g., 500), despite previous records suggesting sensitivity to overhead. The reduction in Python loop iterations from N_trials to 1 per channel is highly effective.
**Action:** Always prioritize vectorizing inner loops in rejection functions that process many epochs.
