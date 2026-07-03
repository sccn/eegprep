## 2025-05-15 - Vectorized griddata_v4 in topoplot
**Learning:** Evaluating biharmonic spline interpolants on a 2D grid using nested Python loops is a major bottleneck. Vectorizing the distance calculation and Green's function evaluation with NumPy broadcasting leads to a ~5.3x speedup for a standard 67x67 grid.
**Action:** Always look for nested loops over coordinate grids when performing spatial interpolation or plotting, and replace them with broadcasting and matrix multiplication.
