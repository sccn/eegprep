# pop_rmbase

Remove baseline means from continuous or epoched EEG data.

`pop_rmbase` subtracts the mean of selected baseline samples from each selected
channel. For continuous data, EEGPrep treats boundary events as data
discontinuities and removes means separately inside each continuous segment.
For epoched data, the selected baseline is removed separately from each epoch.

## Common usage

```python
EEG, com = pop_rmbase(EEG, timerange=[-200, 0], return_com=True)
EEG, com = pop_rmbase(EEG, pointrange=range(1, 51), chanlist=[1, 2], return_com=True)
```

Numeric channel and sample indices are EEGLAB-style 1-based values. Use
`timerange` in the units stored in `EEG["times"]`, which is normally
milliseconds for EEGLAB datasets.
