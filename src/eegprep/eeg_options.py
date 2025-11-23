"""EEG options.

This Python version mirrors the MATLAB key names and default values so you can configure
options in Python pipelines or serialize them to JSON/YAML.
"""

from dataclasses import dataclass, asdict

@dataclass
class EEGOptions:
    """Configuration options for EEG processing, mirroring MATLAB EEGLAB options.

    """

    # STUDY and file options
    option_storedisk: int = 0           # keep at most one dataset in memory
    option_savetwofiles: int = 0        # save header and data as two files
    option_saveversion6: int = 1        # write files in Matlab v6.5 format
    option_saveasstruct: int = 1        # save EEG fields as individual variables
    option_parallel: int = 0            # use parallel toolbox when processing multiple datasets

    # Memory options
    option_single: int = 1              # use single precision in memory
    option_memmapdata: int = 0          # use memory-mapped arrays
    option_eegobject: int = 0           # use EEGLAB EEG object instead of struct

    # ICA options
    option_computeica: int = 1          # precompute ICA activations
    option_scaleicarms: int = 1         # scale ICA components to RMS in microvolt

    # Folder options
    option_rememberfolder: int = 1      # remember last dataset folder

    # Toolbox options
    option_donotusetoolboxes: int = 0   # ignore optional MATLAB toolboxes

    # EEGLAB connectivity and support
    option_showadvanced: int = 0        # show advanced options
    option_boundary99: int = 0          # use type "-99" for boundary events (ERPLAB compatibility)
    option_showpendingplugins: int = 0  # show plugins pending approval
    option_allmenus: int = 0            # show all legacy menu items
    option_htmlingraphics: int = 1      # allow HTML in graphics
    option_checkversion: int = 1        # check for new EEGLAB versions at startup
    option_cachesize: int = 500         # STUDY cache size in MB

    def to_dict(self):
        """Convert the options to a dictionary."""
        return asdict(self)

# Default options instance mirroring the MATLAB file
EEG_OPTIONS = EEGOptions()
EEG_OPTIONS = EEG_OPTIONS.to_dict()
