# POP_IMPORTEGIMAT - Import an EGI Net Station MATLAB export

`pop_importegimat` imports continuous and segmented MATLAB files exported by
EGI Net Station.

Usage:

```python
EEG = pop_importegimat("segmented.mat")
EEG = pop_importegimat("segmented.mat", srate=250, latpoint0=100)
EEG = pop_importegimat("continuous.mat", srate=500, data_field="Session")
EEG, com = pop_importegimat("segmented.mat", return_com=True)
```

Segment variables must be named `<condition>_Segment<number>` and contain
channel-by-sample numeric matrices of identical shape. They are ordered by
condition name and numeric segment number. `latpoint0` gives the time-zero
offset from the start of each segment in milliseconds. A scalar
`samplingRate` variable in the file overrides the `srate` argument; otherwise
`srate` is required.

For continuous exports, `data_field` is `Session` by default and may also be a
variable-name prefix. EGI channel locations are selected from the packaged
montages when the channel count is recognized. Pass `fileloc=""` to leave the
default numbered channel labels in place.

See also: POP_IMPORTDATA, POP_FILEIO, READEGILOCS
