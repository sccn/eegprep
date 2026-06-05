# pop_snapread

Import a SnapMaster `.SMA` recording into an EEG structure.

`pop_snapread(filename, gain)` reads the SnapMaster header and float data,
removes the event channel, applies the gain, and stores detected event-channel
threshold crossings as EEG events. This is implemented for fixture-testable
SnapMaster files and does not require EEGLAB at runtime.

Example:

```python
EEG, com = pop_snapread("recording.SMA", 2.0, return_com=True)
```
