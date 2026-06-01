# eeg_multieegplot

`eeg_multieegplot` opens `eegplot` with current and previous rejection marks
already converted to browser `winrej` rows.

Usage:

```python
window = eeg_multieegplot(data, rej, rejE, oldrej, oldrejE)
model = eeg_multieegplot(data, rej, rejE, oldrej, oldrejE, show=False)
```

For epoched data, `rej` and `oldrej` are trial-length 0/1 vectors and `rejE`
and `oldrejE` are `channels x trials` electrode mark matrices. Previous marks
use EEGLAB's light green color by default; new marks use light blue.
