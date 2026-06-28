## 2025-05-15 - [Broadcasting and Context Manager Optimization in runica]
**Learning:** Manual expansion of vectors for addition (e.g., `bias @ ones(1, M)`) is significantly slower (~3x) than using NumPy broadcasting (`+ bias`). Additionally, wrapping tight loops in context managers like `np.errstate` is more efficient than calling wrappers that use them internally.
**Action:** Always prefer broadcasting over manual matrix-based expansion and minimize context manager entry/exit in high-frequency loops.
