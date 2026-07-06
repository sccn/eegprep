# Bolt's Performance Journal

## 2025-05-15 - Initializing Bolt Journal
**Learning:** Found that several optimizations mentioned in memory are not present in the current codebase state, possibly due to being in different branches or reverted.
**Action:** Re-evaluate and re-apply confirmed optimizations starting with high-impact areas like `topoplot.py` and `runica.py`.

## 2025-05-15 - Vectorizing griddata_v4
**Learning:** The `griddata_v4` function in `topoplot.py` was bottlenecked by a nested Python loop for query point evaluation. Replacing the loop with NumPy broadcasting (`xq + 1j * yq` with shape `(grid, grid, 1)`) and matrix multiplication (`@`) provided a ~4-6x speedup.
**Action:** Always check for nested loops in numerical interpolation or signal processing functions and replace with broadcasting where memory allows.
