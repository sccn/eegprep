# pop_xfirws

Design a windowed-sinc FIR filter and optionally export an xfir-compatible
`.fir` file.

```python
b, a = pop_xfirws(
    srate=500,
    fcutoff=[1, 40],
    ftype="bandpass",
    wtype="hamming",
    forder=330,
)
```

Pass `filename` and `pathname` to write the filter in xfir text format. The
dialog mirrors EEGLAB's FIRFilt helper and includes Kaiser beta estimation,
order estimation, and frequency-response plotting controls.
