
## 2025-05-15 - Vectorized griddata_v4 in topoplot
**Learning:** Replacing a double-nested loop for query point evaluation with NumPy broadcasting and matrix multiplication provides a massive speedup (~11x) for topographic interpolation.
**Action:** Always check for nested loops in interpolation or spatial mapping functions and replace them with vectorized NumPy operations using broadcasting and '@'.
