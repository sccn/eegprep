## 2025-05-15 - Vectorized Diagonal Scaling in Covariance Operations

**Learning:** Matrix operations of the form $V \cdot \text{diag}(D) \cdot V^T$ (common in matrix functions like logm, expm, sqrtm) are significantly more efficient when implemented using NumPy broadcasting (`V * D[..., np.newaxis, :]`) instead of explicit diagonal matrices. This avoids $O(N^3)$ matrix multiplication for the scaling step and eliminates large $O(N^2)$ memory allocations for the diagonal matrices. Additionally, `diag_nd` (creating a batch of diagonal matrices) is much faster using advanced indexing (`res[..., i, i] = M`) than looping and concatenating.

**Action:** Prefer broadcasting for diagonal scaling and advanced indexing for diagonal matrix creation in all performance-sensitive matrix code.
