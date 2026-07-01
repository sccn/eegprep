## 2025-05-15 - [Vectorizing Matrix Power and Fisher-Yates]
**Learning:**
1. In `covariance.py`, creating intermediate diagonal matrices with `diag_nd` for eigenvalue transforms (like `cov_logm`) adds significant overhead compared to using NumPy broadcasting (`V * scale @ Vt`).
2. Even in O(n) loops like Fisher-Yates shuffles, the constant factor of calling a complex rounding function (`round_mat`) and repeatedly calling `stream.rand()` is high. Vectorizing the random number generation and using `math.floor(x + 0.5)` for scalar rounding provides a ~30% speedup.
3. Preserving robustness wrappers like `finite_matmul` is important even when optimizing, to avoid regressions in handling unstable data.

**Action:**
- Prefer broadcasting over `diag_nd` for matrix-diagonal-matrix products.
- Vectorize random number generation outside of tight loops.
- Use `math.floor(x + 0.5)` for fast rounding of non-negative scalars in performance-critical paths.
