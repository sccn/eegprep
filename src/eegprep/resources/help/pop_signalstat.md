POP_SIGNALSTAT - Plot statistics for a channel or component signal.

`pop_signalstat` computes the same core statistic outputs as EEGLAB's
`signalstat`: mean, standard deviation, skewness, excess kurtosis, median,
low/high trim quantiles, trimmed mean, trimmed standard deviation, retained
indices, and a Kolmogorov-Smirnov normality flag.

Examples:

```python
stats = pop_signalstat(EEG, 1, 1, 5)
stats = pop_signalstat(EEG, 0, 2, 10)
```

The GUI asks for the channel/component number and trim percentage, then renders
a histogram, boxplot, QQ plot, topographic context when channel locations are
available, and statistics panel.
