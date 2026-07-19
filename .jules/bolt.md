## 2024-07-17 - [NumPy Advanced Indexing vs Basic Slicing Copy Overhead in Hot Loops]
**Learning:** In hot loops, calling advanced indexing on a large array (e.g., `data[:, timeperm[t:t+block]]`) on every iteration forces NumPy to allocate memory and copy the data. Shuffling the entire array once per epoch (e.g., `shuffled_data = data[:, timeperm]`) and then using basic slicing (e.g., `shuffled_data[:, t:t+block]`) creates zero-copy views and yields massive speedups (over 15-20% overall).
**Action:** Always avoid advanced indexing or random indices inside tight loops by pre-arranging or shuffling arrays outside of the nested block loops.

## 2024-07-17 - [Diagonal Scaling Optimization via Broadcasting]
**Learning:** Multiplying a diagonal matrix with a dense matrix (e.g., `signs_diagonal_matrix @ Y`) incurs unnecessary `O(N^3)` matrix multiplication complexity. By storing the diagonal as a 1D vector and performing row-wise broadcasting (e.g., `signs[:, np.newaxis] * Y`), we achieve the same operation in `O(N^2)` linear element-wise time with zero temporary matrix allocations.
**Action:** Use broadcasting element-wise multiplication for scaling rows/columns by a diagonal instead of converting to/from diagonal matrices via `np.diag`.
