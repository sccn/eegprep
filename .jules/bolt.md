## 2025-05-15 - [Vectorized rmbase]
**Learning:** Reshaping 2D data to (chans, epochs, frames) allows for extremely efficient vectorized baseline removal across all epochs simultaneously using NumPy broadcasting, avoiding expensive Python loops. The performance gain is most significant (~45%) when the number of epochs is large, as it eliminates the per-epoch overhead of slicing and calling `np.nanmean`.
**Action:** Look for opportunities to replace epoch-based loops with (..., epochs, frames) reshapes in other signal processing functions like `eegthresh` or `jointprob`.
