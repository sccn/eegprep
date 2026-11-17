POP_REJMENU - Calculate, inspect, combine, and remove epoch rejection marks.

Usage:

    EEG = pop_rejmenu(EEG)
    EEG, command = pop_rejmenu(EEG, return_com=True)

Use Calc / Plot to calculate threshold, trend, or spectral marks and inspect
them in the scrolling browser. Calculate updates probability or kurtosis marks;
PLOT calculates those marks and opens the same scrolling rejection browser as
Scroll Data or Scroll Acts. Edit limits in the menu and calculate again to
change thresholds.

Close (keep marks) retains calculations and their command history, then combines
checked mark families into `EEG.reject.rejglobal` with `eeg_rejsuperpose`.
Select Reject marked epochs when closing to remove those epochs with
`pop_rejepoch`. Cancel discards changes made in this menu.

Use the scrolling EEGPlot/EEGBrowser buttons to review data or component marks
interactively before updating marks or rejecting epochs.
