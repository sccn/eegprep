# Import BrainVision Analyzer MATLAB data

`pop_loadbva` reads a BrainVision Analyzer MATLAB export (one variable per channel, plus `Channels`, `SampleRate`, `t`, `SegmentCount`, and `Markers`). A `.vhdr` file is imported with `pop_loadbv` instead.

The File menu records `EEG = pop_loadbva('file.mat');`.
