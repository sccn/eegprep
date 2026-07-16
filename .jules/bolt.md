## 2026-07-16 - [Topoplot Vectorization]
**Learning:** Replacing double-nested loops over a 2D grid with NumPy broadcasting (3D array expansion) and matrix multiplication (@) provides a significant (3.8x - 4.2x) speedup for biharmonic spline interpolation.
**Action:** Always look for nested loops over query grids in spatial interpolation functions and replace them with vectorized broadcasting.
