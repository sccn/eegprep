## 2026-07-08 - [Optimization of topoplot grid interpolation]
**Learning:** Replaced a double-nested loop in `griddata_v4` with vectorized NumPy broadcasting and matrix multiplication (@). This is particularly effective for 2D grid evaluations in interpolation functions where query points are numerous.
**Action:** Always look for nested loops over query grids in signal processing and visualization functions and replace them with 3D broadcasting + @ to shift computation to BLAS.
