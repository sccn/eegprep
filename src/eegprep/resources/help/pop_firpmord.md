# pop_firpmord

Estimate Parks-McClellan FIR order and pass/stop weights for `pop_firpm`.

```python
m, wtpass, wtstop = pop_firpmord([0, 40, 48, 125], [1, 0], [0.01, 0.001], 250)
```

Programmatic calls accept frequency band edges, desired band amplitudes,
allowable deviations, and an optional sampling rate. The GUI path accepts
passband ripple and stopband attenuation in dB and converts them to the
deviation values used by the order estimator.
