## 2025-05-15 - Vectorized Preprocessing and Optimized Training Loops in runica.py

**Learning:** Moving `np.errstate` context managers outside of tight iterative loops and replacing internal helper wrappers (like `_matmul`) with native operators (like `@`) eliminates significant function call and context management overhead. Additionally, using NumPy broadcasting instead of explicit matrix multiplication with ones (e.g., `+ bias` instead of `_matmul(bias, onesrow)`) avoids unnecessary (N \times M)$ operations.

**Action:** Always look for repeated context management or small helper calls inside loops. Use broadcasting for bias/offset addition instead of matrix products.
