"""EEG rejection functions."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .utils.misc import round_mat


def _is_boundary_event(event: Dict) -> bool:
    t = event.get("type")
    if isinstance(t, str):
        return t.lower() == "boundary"
    if isinstance(t, (int, float)):
        try:
            return int(t) == -99
        except Exception:
            return False
    return False


def eegrej(indata, regions, timelength, events: Optional[List[Dict]] = None) -> Tuple[np.ndarray, float, List[Dict], np.ndarray]:
    """Remove [beg end] sample ranges (1-based, inclusive) from continuous data and
    update events.

    Parameters
    ----------
    indata : array-like
        2D array shaped (channels, frames)
    regions : array-like
        Shape (n_regions, 2), 1-based [beg end] per row
    timelength : float
        Total duration of the original data in seconds
    events : list of dict, optional
        List of dicts with at least key 'latency'; optional keys include 'type' and 'duration'.
        If None or empty, boundary events will still be inserted based on regions.

    Returns
    -------
    outdata : ndarray
        Data with columns removed
    newt : float
        New total time in seconds
    events_out : list of dict
        Updated events list of dictionaries (with inserted boundaries)
    boundevents : ndarray
        Boundary latencies (float, 1-based, with +0.5 convention)
    """
    x = np.asarray(indata)
    if x.ndim != 2:
        raise ValueError("indata must be 2D (channels, frames)")
    n = x.shape[1]

    r = np.asarray(regions, dtype=float)
    if r.size == 0:
        # nothing to remove; still ensure events sorted and valid
        events_out = [] if events is None else [dict(ev) for ev in events]
        # Sort events by latency if present
        if events_out and all("latency" in ev for ev in events_out):
            events_out.sort(key=lambda ev: ev.get("latency", float("inf")))
        boundevents = np.array([], dtype=float)
        return x, float(timelength), events_out, boundevents

    if r.ndim != 2 or r.shape[1] != 2:
        raise ValueError("regions must be of shape (n_regions, 2)")

    # Round, clamp to [1, n], sort each row then sort rows (EEGLAB parity)
    r = np.rint(r).astype(int)
    r[:, 0] = np.clip(r[:, 0], 1, n)
    r[:, 1] = np.clip(r[:, 1], 1, n)
    r.sort(axis=1)
    r = r[np.lexsort((r[:, 1], r[:, 0]))]

    # Enforce non-overlap by shifting starts forward (like MATLAB)
    for i in range(1, r.shape[0]):
        if r[i - 1, 1] >= r[i, 0]:
            r[i, 0] = r[i - 1, 1] + 1
    # Drop empty or inverted regions after adjustment
    r = r[r[:, 0] <= r[:, 1]]
    if r.size == 0:
        events_out = [] if events is None else [dict(ev) for ev in events]
        if events_out and all("latency" in ev for ev in events_out):
            events_out.sort(key=lambda ev: ev.get("latency", float("inf")))
        boundevents = np.array([], dtype=float)
        return x, float(timelength), events_out, boundevents

    # Build reject mask (convert 1-based to 0-based slices)
    # MATLAB: reject(beg:end) = 1  (includes both beg and end, 1-based)
    # Python: reject[beg-1:end] = True  (includes beg-1 to end-1, since end is exclusive in Python slicing)
    # To match MATLAB's inclusive end, we need reject[beg-1:end] where end is inclusive
    reject = np.zeros(n, dtype=bool)
    for beg, end in r:
        reject[beg - 1:end] = True  # This matches MATLAB reject(beg:end) when end is already the inclusive end

    # Prepare events
    ori_events: List[Dict] = [] if events is None else [dict(ev) for ev in events]
    events_out: List[Dict] = [dict(ev) for ev in ori_events]

    # Recompute event latencies (if events have 'latency') and remove events strictly inside regions
    if events_out and all("latency" in ev for ev in events_out):
        ori_lat = np.array([float(ev.get("latency", float("nan"))) for ev in events_out], dtype=float)
        lat = ori_lat.copy()
        rejected_per_region: List[List[int]] = []
        for beg, end in r:
            # indices strictly inside (beg, end)
            rej_idx = np.where((ori_lat > beg) & (ori_lat < end))[0].tolist()
            rejected_per_region.append(rej_idx)
            # subtract span from latencies whose original latency is strictly after region start
            span = int(end - beg + 1)
            lat[ori_lat > beg] -= span

        # Apply updated latencies
        for i, ev in enumerate(events_out):
            ev["latency"] = float(lat[i])

        # Remove events inside rejected regions
        rm_idx = sorted(set(idx for group in rejected_per_region for idx in group))
        if rm_idx:
            keep_mask = np.ones(len(events_out), dtype=bool)
            keep_mask[rm_idx] = False
            events_out = [ev for j, ev in enumerate(events_out) if keep_mask[j]]

    # Boundary latencies: start-1, then subtract cumulative prior durations, then +0.5
    base_durations = (r[:, 1] - r[:, 0] + 1).astype(int)

    # If we have original events and they include type/duration, add nested boundary durations
    durations = base_durations.astype(float).copy()
    if ori_events and all("latency" in ev for ev in ori_events):
        ori_lat = np.array([float(ev.get("latency", float("nan"))) for ev in ori_events], dtype=float)
        for i_region, (beg, end) in enumerate(r):
            inside_mask = (ori_lat > beg) & (ori_lat < end)
            selected_events = [ori_events[i] for i, m in enumerate(inside_mask) if m]
            extra = 0.0
            for ev in selected_events:
                if _is_boundary_event(ev):
                    extra += float(ev.get("duration", 0.0) or 0.0)
            durations[i_region] += extra

    # Compute boundevents considering prior removals
    boundevents = r[:, 0].astype(float) - 1.0
    if len(durations) > 1:
        cums = np.concatenate([[0.0], np.cumsum(durations[:-1])])
        boundevents = boundevents - cums
    boundevents = boundevents + 0.5
    boundevents = boundevents[boundevents >= 0]

    # Excise samples
    newx = x[:, ~reject]
    newn = int(newx.shape[1])

    # Update total time proportionally
    newt = float(timelength) * (newn / float(n))

    # Remove boundary events that would fall exactly after the last sample + 0.5
    boundevents = boundevents[boundevents < (newn + 1)]

    # Merge duplicate boundary latencies and sum durations for duplicates
    if boundevents.size:
        rounded = round_mat(boundevents, 12)
        merged_be: List[float] = []
        merged_du: List[float] = []
        for i, be in enumerate(rounded):
            if not merged_be:
                merged_be.append(be)
                merged_du.append(float(durations[i]))
            else:
                if np.isclose(be, merged_be[-1]):
                    merged_du[-1] += float(durations[i])
                else:
                    merged_be.append(be)
                    merged_du.append(float(durations[i]))
        boundevents = np.asarray(merged_be, dtype=float)
        durations = np.asarray(merged_du, dtype=float)
    else:
        durations = np.asarray([], dtype=float)

    # Insert boundary events into events list only if input events were provided
    if ori_events:
        bound_type = "boundary"
        for i in range(len(boundevents)):
            be = float(boundevents[i])
            if be > 0 and be < (newn + 1):
                events_out.append({
                    "type": bound_type,
                    "latency": be,
                    "duration": float(durations[i] if i < len(durations) else (base_durations[i] if i < len(base_durations) else 0.0)),
                })

    # Remove events with latency out of bound (> newn+1)
    filtered: List[Dict] = []
    for ev in events_out:
        latv = float(ev.get("latency", float("inf")))
        if latv <= (newn + 1):
            filtered.append(ev)
    events_out = filtered

    # Sort by latency
    events_out.sort(key=lambda ev: ev.get("latency", float("inf")))

    # Handle contiguous boundary events with same latency: merge durations
    if events_out:
        merged_events: List[Dict] = []
        for ev in events_out:
            if merged_events and _is_boundary_event(ev) and _is_boundary_event(merged_events[-1]) \
               and np.isclose(float(ev.get("latency", 0.0)), float(merged_events[-1].get("latency", 0.0))):
                prev_dur = float(merged_events[-1].get("duration", 0.0) or 0.0)
                cur_dur = float(ev.get("duration", 0.0) or 0.0)
                merged_events[-1]["duration"] = prev_dur + cur_dur
            else:
                merged_events.append(ev)
        events_out = merged_events

    return newx, newt, events_out, boundevents