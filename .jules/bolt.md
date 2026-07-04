## 2025-05-15 - [Vectorized griddata_v4 query evaluation]
**Learning:** Replacing a double-nested loop for query point evaluation in biharmonic spline interpolation with NumPy broadcasting and matrix multiplication yielded a ~6x speedup. The memory trade-off for a 3D distance matrix is negligible for typical EEG topoplot grid sizes.
**Action:** Use broadcasting and matrix multiplication to eliminate nested loops in spatial interpolation or query-point evaluation tasks.
