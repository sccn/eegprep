# POP_LIMO - LIMO limitation

EEGPrep does not implement EEGLAB's external LIMO toolbox workflow.

Calling `pop_limo` raises a clear `NotImplementedError` instead of creating
placeholder LIMO files or pretending external MATLAB behavior is available.
Use EEGPrep's standalone statistics helpers and `std_limodesign` for
in-package analyses and LIMO-compatible design matrices, or run LIMO in
EEGLAB/MATLAB when you need the external LIMO model-fitting and result
browsing workflow.

See also: POP_LIMORESULTS, STD_LIMODESIGN, EEG_HELPSTATISTICS
