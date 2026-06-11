# pop_firwsord

Estimate an even windowed-sinc FIR filter order for a requested transition
bandwidth.

```python
m = pop_firwsord("hamming", 500, 2)
m, dev = pop_firwsord("kaiser", 500, 2, 0.001, return_dev=True)
```

Supported windows are `rectangular`, `hann`, `hamming`, `blackman`, and
`kaiser`. Kaiser designs require the maximum passband deviation. The dialog
enables the deviation field only when Kaiser is selected, matching EEGLAB's
workflow.
