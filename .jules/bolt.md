## 2025-05-14 - Optimized Covariance Matrix Operations

**Learning:** Matrix power functions (logm, expm, powm, sqrtm) implemented via eigen-decomposition can be significantly optimized by replacing diagonal matrix creation and full matrix multiplication with NumPy broadcasting. Scaling eigenvectors by eigenvalues is $O(N^2)$ vs $O(N^3)$ for matrix multiplication, and avoids $O(N^2)$ space for the diagonal matrix. `diag_nd` was also a bottleneck due to loop-based concatenation.

**Action:** Prefer broadcasting for scaling columns/rows of matrices by vectors. Use advanced indexing for batch diagonal matrix creation.
