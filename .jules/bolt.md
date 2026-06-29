## 2025-05-15 - Vectorized griddata_v4 in topoplot.py
**Learning:** Vectorizing the biharmonic spline interpolation in `griddata_v4` by replacing a double-nested loop for query point evaluation with NumPy broadcasting and matrix multiplication (`@`) yields a ~6.6x speedup.
**Action:** Always check for opportunities to replace explicit spatial evaluation loops with vectorized distance calculations using broadcasting (e.g., `xy[:, np.newaxis] - query_points[np.newaxis, :]`) and matrix multiplication.
