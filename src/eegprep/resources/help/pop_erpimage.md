# pop_erpimage

Plots an event-related potential (ERP) image of one channel
or one independent component analysis (ICA) component:
single trials stacked as rows, sorted by an event value and smoothed across trials,
with the ERP, the average of the trials, underneath.

```python
result, com = pop_erpimage(EEG, 1, 14, return_com=True)
```

The arguments follow EEGLAB's order:

```python
pop_erpimage(EEG, typeplot, channel, projchan, titleplot, smooth, decimate,
             sortingtype, sortingwin, sortingeventfield, *erpimage_options)
```

- `typeplot`: 1 for channels, 0 for ICA components.
  The dataset must be epoched, with more than one trial.
- `channel`: 1-based channel numbers or labels (`"Cz"`), or component numbers.
  Ranges such as `"3:5"` work, and labels ignore case.
  Several channels are averaged.
- `projchan`: channels to back-project the components onto, in microvolts.
  Several components need a projection channel and are summed,
  several projection channels are averaged,
  and the scalp map shows the mean of the component maps.
  Channel plots ignore `projchan`.
- `titleplot`: the figure title.
  By default it is built as EEGLAB builds it:
  the channel labels joined without a separator (`F3Fz`),
  the component numbers (`Comp. 6`), or `Comp. 6 -> Chan. 27` for a projection.
- `smooth` (default 5): the moving-average width in trials;
  0 and 1 show every trial unchanged.
- `decimate` (default 0): the window advance in trials.
  0 and 1 both advance one trial,
  and a value above the square root of the number of trials gives that many image rows instead.
- `sortingtype`, `sortingwin`, `sortingeventfield`:
  sort the trials by an event field, for example the `"latency"` of `"rt"` events,
  within a time window `[start end]` in milliseconds
  (EEGLAB's help says seconds, but its event lookup uses milliseconds).
  Text values sort alphabetically.
  When an epoch has several matching events,
  the first latency or the last value of another field is used,
  and several latencies in one epoch are reported in a warning.
  Epochs without a matching event are left out with a warning that lists them;
  when no epoch matches, `pop_erpimage` raises an error naming the types and the window.
  Without a field the trials keep their order,
  and `sortingtype` or `sortingwin` alone are ignored with a warning.
  A string `sortingtype` is one event type, even with spaces (`"S  1"`).
  Numeric event types match by value, so `2`, `2.0` and `"2"` all select events of type 2.
  Sorting by `urevent` uses EEGPrep's 0-based `urevent` values,
  one less than the numbers EEGLAB shows.

Any `erpimage` option can follow, as EEGLAB tokens or as keywords.
For example, `phasesort=[0, 25, 10]` sorts the trials by phase,
`coher=[10, 10, 0.01]` adds amplitude and coherence traces,
`align=float('inf')` aligns the trials on the median event value,
`renorm='yes'` rescales the event values to the epoch,
`nosort=True` keeps the trial order but plots the event values,
and `noplot=True` sorts without drawing the values.
`help(erpimage)` and the API reference list every option.

EEGLAB tokens such as `'phasesort', [0, 25, 10]` can follow only after all ten positional arguments;
otherwise, use keywords.
A lowercase option name in the place of `projchan` or `titleplot`,
or a title that is not text, raises a `ValueError` instead of being misread.
When an option is given twice, the later one wins, as in EEGLAB.

`limits=[tmin, tmax, erpmin, erpmax, ampmin, ampmax, cohmin, cohmax, basedB]`
sets the time limits (ms), the ERP limits (microvolts), the amplitude limits (dB),
the coherence limits and the baseline amplitude (dB), as the dialog's limit fields do;
a NaN keeps that default.
`rng=` seeds the coherence bootstrap and the `erpalpha` permutations;
only an integer seed is written to the history command, so use one to replay the same values.

`pop_erpimage` draws the ERP trace, the colorbar and a scalp map by default,
also from the command line, where EEGLAB draws them only from its dialog.
Use `erp=False`, `cbar=False` or `plotmap=False` (or `'off'`) to leave them out.
The scalp map marks the channel location, or shows the component map.
It is left out, with a warning, when several channels are averaged or the channels have no locations,
and a `topo=` option replaces it.
The function returns the `erpimage` result
(`result.outdata`, `result.sortidx`, `result.figure`, ...).

```python
result, com = pop_erpimage(
    EEG, 1, "POz", None, "", 10, 1, ["rt"], [0, 1000], "latency",
    "phasesort", [0, 25, 10], return_com=True,
)
```

## Dialog

`Plot > Channel ERP image` and `Plot > Component ERP image` open the ERP image dialog.
Smoothing starts at the number of trials minus 5, kept between 0 and 10,
and Downsampling at 1, as in EEGLAB.
`Channel`, `Smoothing` and `Downsampling` cannot be left blank.

Dialog fields:

- Channel or Component(s), and its `...` list: `channel`.
  For components, the `...` button next to `Project to channel #` picks the channel
  (EEGLAB disables it).
- Project to channel #: `projchan`.
- Fig. title: `titleplot`.
- Smoothing and Downsampling: `smooth` and `decimate`.
- Plot scalp map, Plot ERP, Plot colorbar: `plotmap`, `erp` and `cbar`.
- ERP limits, Time limits (ms), Amplitude limits (dB), Coher limits and Baseline ampl. (dB):
  the parts of `limits`.
  The time limits are passed only when you change them.
- Color limits: `caxis`, `[low high]` or a fraction of the data range.
- Epoch-sorting field and Event type(s): `sortingeventfield` and `sortingtype`.
  Their buttons list the dataset's event fields and event types;
  the type list is sorted as EEGLAB's `unique()` sorts it (numerically for numeric types),
  and the types you pick replace the text in the box.
  Each button reminds you to fill the other box.
  Quote typed types that contain spaces, for example `'S  1' 'S  2'`.
- Event time range: `sortingwin`, exactly two values in ms.
- Rescale: `renorm` (`yes`, `no`, or a formula such as `3*x+2`);
  it applies only with a sorting field.
- Align: `align`, a latency in ms, or `Inf` for the median event value.
- Don't sort by value and Don't plot values: `nosort` and `noplot`.
- Sort trials by phase: Frequency (Hz, or minHz maxHz), Percent low-amp. trials to ignore
  and Window center (ms) fill `phasesort=[center, percent, frequencies]`;
  the wavelet has 3 cycles (set `cycles` in More options to change it).
- Inter-trial coherence: Frequency and Signif. level fill `coher=[frequencies, alpha]`,
  and Image amps sets `plotamps`.
  The phase-sorting and coherence frequencies mirror each other when both are filled.
- Plot spectrum (minHz maxHz): `spec`.
- Mark times (ms): `vert`.
- More options: any other `erpimage` option, as `'key', value` pairs or `key=value`.

As in EEGLAB, More options are read first and every field you fill replaces them;
an empty field or unchecked box leaves a More options value in place,
and the ERP axis label always comes from the dialog.

The menu reopens the dialog with the last matching command in `ALLCOM`, as EEGLAB does.
From the console, pass that command as `lastcom=`,
for example `pop_erpimage(EEG, 1, lastcom=LASTCOM)` when the last command was an ERP image;
any other command raises a `ValueError`, and so do positional arguments or `erpimage` options passed with it.
The `typeplot` argument wins over the one inside `lastcom`.
Or pass arguments with `gui=True` to open the dialog filled with them.
Options without a field of their own appear in More options
(EEGLAB leaves that box empty).
The dialog's values replace the arguments;
options the dialog cannot show, such as a `topo` map with its own channel locations,
are applied as given.

## Differences from EEGLAB

- `lastcom` is a keyword; EEGLAB passes it in place of `channel`,
  and a text `channel` here is a channel label.
- The ERP trace, colorbar and scalp map are on from the command line too.
- The dialog's Event time range takes exactly two values.
- A `topo` map does not carry EEGLAB's `chaninfo`,
  and the figure keeps Matplotlib's window name and colors.

The command recorded in history (`LASTCOM`) is a Python call that you can edit
and run in `eegprep-console`. Making an ERP image does not change the dataset.

When scripting, pass `plot='off'` to build the figure without opening a window.
Pass `target=` a Matplotlib Figure or SubFigure to draw into it.
