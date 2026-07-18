# Bolt's Journal - Critical Learnings Only

## 2025-02-17 - Vectorizing `griddata_v4` in Topoplot
**Learning:** Bi-harmonic spline interpolation in `topoplot.py` (`griddata_v4`) was bottlenecked by a double-nested Python loop over the query coordinates grid (`m x n`). Since `GRID_SCALE` is typically small/medium (e.g., 67x67), vectorizing via 3D broadcasting and a single `@` matrix multiplication has very low memory overhead and achieves massive speedups.
**Action:** Replace the loop over grid rows and columns with 3D broadcasting `np.abs(q[:, :, None] - xy[None, None, :])` and matrix-multiply with `weights`.
