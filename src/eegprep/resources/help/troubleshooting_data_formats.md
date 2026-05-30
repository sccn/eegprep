# TROUBLESHOOTING_DATA_FORMATS - Troubleshooting data formats

EEGPrep can import EEGLAB `.set` files, MATLAB arrays, text/NumPy arrays,
EDF/BDF/GDF files, BrainVision headers, EGI MFF folders, Neuroscan CNT/EEG
files, and BIDS EEG datasets when the required Python dependencies are
installed.

If import fails:

- Confirm the file extension matches the selected menu action.
- Use `pop_loadset` for EEGLAB `.set` files.
- Use `pop_importdata` for numeric arrays and text/NumPy/MATLAB array files.
- Use `pop_fileio` or the File-IO plugin menu items for EDF/BDF/GDF,
  BrainVision, MFF, CNT, and EEG formats.
- Use `pop_importbids` for BIDS folders or supported BIDS EEG files.
- Check that optional readers such as MNE and pyEDFlib are installed in the
  environment used to launch EEGPrep.

See also: POP_LOADSET, POP_IMPORTDATA, POP_FILEIO, POP_BIOSIG, POP_IMPORTBIDS
