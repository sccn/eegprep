## 2025-03-05 - [Grid evaluation vectorization in topoplot]
**Learning:** Nested Python loops over coordinate meshes (`xq`, `yq`) are extremely slow when executing NumPy calculations inside, especially for mathematical formulas like biharmonic spline interpolation. Evaluating elements individually causes substantial overhead. Vectorizing coordinate meshes using complex numbers and broadcasting enables fast operations at scale.
**Action:** Always refactor meshgrid-based spatial evaluations to use complex 1D/2D broadcasting and matrix multiplication with `@`.
