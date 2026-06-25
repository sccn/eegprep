## 2025-05-15 - Vectorized RNG and math.floor in tight loops
**Learning:** Replacing scalar `stream.rand()` calls with a single vectorized `stream.rand(n)` call and using `math.floor(x + 0.5)` instead of `round_mat` (which has higher overhead) significantly improves performance in tight loops like Fisher-Yates shuffles (approx. 25-40% speedup) while maintaining exact RNG parity.
**Action:** Use vectorized RNG pre-generation and `math.floor` for scalar rounding in performance-critical numerical loops.
