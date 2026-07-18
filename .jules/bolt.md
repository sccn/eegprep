# Bolt's Performance Journal

## 2025-01-30 - Vectorizing Joint Probability
**Learning:** Vectorizing the inner trial loop of joint probability calculations reduces function call and slicing overhead significantly (approx. 23% speedup). Reshaping 1D probability arrays to 2D allows standard NumPy sum reductions.
**Action:** Avoid nested loops over trials when computing joint probabilities by flattening and using vectorized operations along `axis=1`.
