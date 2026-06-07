# vis_artifacts

Compare a clean_rawdata output dataset with its original input and highlight
rejected samples in the EEG browser.

```python
diag = vis_artifacts(clean_eeg, original_eeg, show=False)
vis_artifacts(clean_eeg, original_eeg)
```

When `show=False`, the function returns diagnostics without opening Qt. The
diagnostics include rejected sample intervals, rejected fraction, removed
channel indices and labels, and the `winrej` matrix that would be passed to
`eegplot`.
