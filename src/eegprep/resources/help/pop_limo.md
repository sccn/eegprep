# POP_LIMO - LIMO limitation

EEGPrep does not implement EEGLAB's external LIMO toolbox workflow.

Calling `pop_limo` raises a clear `NotImplementedError` instead of creating
placeholder LIMO files or pretending external MATLAB behavior is available.
Use EEGPrep's standalone statistics helpers for in-package analyses, or run
LIMO in EEGLAB/MATLAB and import explicit results through your own analysis
code.

See also: POP_LIMORESULTS, EEG_HELPSTATISTICS
