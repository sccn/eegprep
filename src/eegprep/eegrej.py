import numpy as np

def eegrej(indata, regions, timelength, eventlatencies=None):
    """
    Remove [beg end] sample ranges (1-based, inclusive) from continuous data.

    Inputs
      indata: 2D array shaped (channels, frames)
      regions: array-like with shape (n_regions, 2), 1-based [beg end] per row
      timelength: total duration of the original data in seconds
      eventlatencies: iterable of event latencies in samples (1-based). Optional.

    Returns
      outdata: data with columns removed
      newt: new total time in seconds
      newevents: adjusted event latencies (NaN for events inside removed regions)
      boundevents: boundary latencies (float, 1-based, with +0.5 convention)
    """
    x = np.asarray(indata)
    if x.ndim != 2:
        raise ValueError("indata must be 2D (channels, frames)")
    n = x.shape[1]

    r = np.asarray(regions, dtype=float)
    if r.size == 0:
        # nothing to remove
        newx = x
        newt = float(timelength)
        if eventlatencies is None:
            newevents = None
        else:
            newevents = np.asarray(eventlatencies, dtype=float)
        boundevents = np.array([], dtype=float)
        return newx, newt, newevents, boundevents

    if r.ndim != 2 or r.shape[1] != 2:
        raise ValueError("regions must be of shape (n_regions, 2)")

    # Round, clamp to [1, n], sort each row then sort rows
    r = np.rint(r).astype(int)
    r[:, 0] = np.clip(r[:, 0], 1, n)
    r[:, 1] = np.clip(r[:, 1], 1, n)
    r.sort(axis=1)
    r = r[np.lexsort((r[:, 1], r[:, 0]))]

    # Enforce non-overlap by shifting starts forward
    for i in range(1, r.shape[0]):
        if r[i - 1, 1] >= r[i, 0]:
            r[i, 0] = r[i - 1, 1] + 1
    # Drop empty or inverted regions after adjustment
    r = r[r[:, 0] <= r[:, 1]]
    if r.size == 0:
        newx = x
        newt = float(timelength)
        if eventlatencies is None:
            newevents = None
        else:
            newevents = np.asarray(eventlatencies, dtype=float)
        boundevents = np.array([], dtype=float)
        return newx, newt, newevents, boundevents

    # Build reject mask (convert 1-based to 0-based slices)
    reject = np.zeros(n, dtype=bool)
    for beg, end in r:
        reject[beg - 1:end] = True

    # Recompute event latencies
    if eventlatencies is None:
        newevents = None
        rejected_events_masks = None
    else:
        ev = np.asarray(eventlatencies, dtype=float).copy()
        newevents = ev.copy()
        durations = (r[:, 1] - r[:, 0] + 1)
        # Mark events inside any region as NaN
        inside = np.zeros(ev.shape, dtype=bool)
        for beg, end in r:
            inside |= (ev >= beg) & (ev <= end)
        newevents[inside] = np.nan
        # Shift remaining events left by total removed samples preceding them
        # Use original ev for comparisons, per EEGLAB behavior
        for beg, end in r:
            span = end - beg + 1
            affected = ev > end  # original latency strictly after the region
            newevents[affected] -= span

    # Boundary latencies: start-1, then account for prior removals, then +0.5
    durations = (r[:, 1] - r[:, 0] + 1).astype(int)
    boundevents = r[:, 0].astype(float) - 1.0
    # subtract cumulative durations of earlier regions
    cums = np.concatenate([[0], np.cumsum(durations[:-1])]).astype(float)
    boundevents = boundevents - cums
    boundevents = boundevents + 0.5
    boundevents = boundevents[boundevents >= 0]

    # Excise samples
    newx = x[:, ~reject]
    newn = newx.shape[1]

    # Update total time proportionally
    newt = float(timelength) * (newn / float(n))

    # Remove boundary events that would fall exactly after the last sample + 0.5
    boundevents = boundevents[boundevents < (newn + 1)]

    # Merge duplicate boundary latencies (rare after de-overlap, but keep parity with EEGLAB)
    if boundevents.size:
        be = boundevents
        # Since we do not track duration objects here, just unique latencies preserving order
        _, idx = np.unique(np.round(be, 12), return_index=True)
        boundevents = be[np.sort(idx)]

    return newx, newt, newevents, boundevents