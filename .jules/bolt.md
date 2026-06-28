## 2026-06-28 - Vectorizing Bi-harmonic Spline Evaluation in topoplot
**Learning:** Nested Python loops in `griddata_v4` for evaluating query points were a major bottleneck. Repeatedly entering/exiting `np.errstate` context managers inside these loops added further overhead.
**Action:** Use NumPy broadcasting and the `@` operator to vectorize evaluation across the entire query grid at once. Move context managers outside the core calculation.
