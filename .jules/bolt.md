## 2025-05-15 - Optimized runica.py with focused diff

**Learning:** Replacing matrix-based bias addition (`_matmul(bias, onesrow)`) with NumPy broadcasting (`+ bias`) significantly reduces (N \times M)$ overhead and memory allocations in tight iterative loops. Using the native `@` operator and pre-configuring `np.seterr` outside of loops provides further speedups by reducing function call and context management overhead while keeping the diff clean.

**Action:** Prefer broadcasting for bias/offset addition. Use `np.seterr` globally or at the function level for expected numerical warnings in iterative algorithms to avoid re-indenting loop bodies with `with np.errstate`.
