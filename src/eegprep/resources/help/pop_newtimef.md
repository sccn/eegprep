POP_NEWTIMEF - Plot channel or component time-frequency decomposition.

`pop_newtimef` computes an event-related spectral perturbation (ERSP) image and
inter-trial coherence (ITC) image for one channel or ICA component. When channel
locations are available, the figure adds a scalp-map inset (the channel's head
location, or the component's interpolated map) and a caption naming the channel
or component.

Examples:

```python
result = pop_newtimef(EEG, 1, 1, [-100, 200], [3, 0.8])
result = pop_newtimef(EEG, 0, 2, [-100, 200], [0], freqs=[4, 30], padratio=2)
```

The EEGPrep implementation provides deterministic FFT output for `cycles=[0]`
and Morlet wavelet output for non-zero `cycles`, with EEGLAB-compatible inputs,
history strings, baseline modes, bootstrap significance outputs, and image or
curve plotting.

Time-warped event markers can be supplied with `timewarp`, a trials-by-events
matrix of event latencies in milliseconds. Use `timewarpms` to supply the common
reference latencies and `timewarpidx` to choose which event columns are marked
with vertical lines. If `timewarpms` is omitted, EEGPrep uses the median latency
of each event column, matching EEGLAB's standalone workflow.

The "ERSP color limits" and "ITC color limits" dialog fields (options `erspmax`
and `itcmax`) set the image color-scale maxima. `erspmax` is applied symmetrically
as `[-erspmax, erspmax]`; `itcmax` sets the ITC image upper limit. Leave a field
empty to auto-scale.

The "plot ITC phase" checkbox (option `plotphase`) colors the ITC image by the sign
of the coherence phase; leave it unchecked to show coherence magnitude only. Scripts
can also pass `plotphaseonly='on'` to display the phase angle in degrees, or
`pcontour='on'` to outline significant regions with contours instead of masking them.

The `tf cycle calc` button opens the Morlet wavelet cycle calculator. It writes
the calculated frequency and cycle vectors back to the parent `pop_newtimef`
dialog.

When scripting, pass `plot='off'` to build the figure without opening a window; the default `plot='on'` displays it.
