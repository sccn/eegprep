## 2025-05-15 - Numerical Optimization Patterns

**Learning:**
1. In `_interpMx` (spherical interpolation), avoiding redundant array copies (`G.copy()`) and replacing `np.all(dG < tol)` with `dG.max() < tol` for non-negative convergence metrics reduced iteration overhead significantly (~25% speedup). Creating boolean masks for `np.all` is expensive in tight loops.
2. Vectorizing random number generation in Fisher-Yates shuffles (using `stream.rand(n)` once instead of `stream.rand()` in every iteration) and using `math.floor(x + 0.5)` for scalar index rounding provides ~40% speedup while maintaining bit-perfect MATLAB parity.

**Action:**
- Audit tight numerical loops for redundant `.copy()` calls and intermediate array allocations.
- Prefer `max()` comparisons over `all()` for non-negative convergence checks.
- Pre-generate random sequences for shuffles or sampling loops.
