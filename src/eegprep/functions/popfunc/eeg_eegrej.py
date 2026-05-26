"""EEG data rejection functions."""

from typing import List, Dict, Optional, Tuple
import numpy as np
from copy import deepcopy
from ..miscfunc.misc import round_mat


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


def _eegrej(
    indata, regions, timelength, events: Optional[List[Dict]] = None
) -> Tuple[np.ndarray, float, List[Dict], np.ndarray]:
    """Remove [beg end] sample ranges (1-based, inclusive) from continuous data and update events.

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
        reject[beg - 1 : end] = True  # This matches MATLAB reject(beg:end) when end is already the inclusive end

    # Prepare events
    ori_events: List[Dict] = [] if events is None else [dict(ev) for ev in events]
    events_out: List[Dict] = [dict(ev) for ev in ori_events]

    # Recompute event latencies and remove events inside regions.
    # MATLAB eeg_eegrej.m uses inclusive bounds (>= and <=) but preserves
    # boundary events (line 117: allEventFlag(boundaryIndices) = false).
    if events_out and all("latency" in ev for ev in events_out):
        ori_lat = np.array([float(ev.get("latency", float("nan"))) for ev in events_out], dtype=float)
        lat = ori_lat.copy()
        rejected_per_region: List[List[int]] = []
        for beg, end in r:
            # Inclusive bounds matching MATLAB (>= beg & <= end)
            rej_idx = np.where((ori_lat >= beg) & (ori_lat <= end))[0].tolist()
            # Preserve boundary events inside regions (MATLAB line 117)
            rej_idx = [i for i in rej_idx if not _is_boundary_event(ori_events[i])]
            rejected_per_region.append(rej_idx)
            # subtract span from latencies whose original latency is strictly after region start
            span = int(end - beg + 1)
            lat[ori_lat > beg] -= span

        # Apply updated latencies
        for i, ev in enumerate(events_out):
            ev["latency"] = float(lat[i])

        # Remove non-boundary events inside rejected regions
        rm_idx = sorted(set(idx for group in rejected_per_region for idx in group))
        if rm_idx:
            keep_mask = np.ones(len(events_out), dtype=bool)
            keep_mask[rm_idx] = False
            events_out = [ev for j, ev in enumerate(events_out) if keep_mask[j]]

    # Boundary latencies: start-1, then subtract cumulative prior durations, then +0.5
    base_durations = (r[:, 1] - r[:, 0] + 1).astype(int)

    # Find nested boundary events inside regions and accumulate their durations.
    # MATLAB eeg_insertbound uses findnested() with inclusive bounds to find
    # boundary events inside each region, adds their durations to the new
    # boundary's .duration field, and removes the nested events.
    durations = base_durations.astype(float).copy()
    nested_to_remove: List[int] = []  # indices into events_out to remove after loop
    if ori_events and all("latency" in ev for ev in ori_events):
        ori_lat = np.array([float(ev.get("latency", float("nan"))) for ev in ori_events], dtype=float)
        for i_region, (beg, end) in enumerate(r):
            # Inclusive bounds matching MATLAB findnested (> beg & < end for strict interior)
            inside_mask = (ori_lat > beg) & (ori_lat < end)
            extra = 0.0
            for i_ev, m in enumerate(inside_mask):
                if m and _is_boundary_event(ori_events[i_ev]):
                    extra += float(ori_events[i_ev].get("duration", 0.0) or 0.0)
                    # Mark for removal from events_out (find by matching latency)
                    nested_to_remove.append(i_ev)
            durations[i_region] += extra

    # Compute boundevents considering prior removals.
    # Use base_durations (raw region widths) for latency subtraction, matching
    # MATLAB eeg_insertbound which uses lengths = regions(:,2)-regions(:,1)+1.
    # The durations array (which includes nested boundary durations) is only
    # used for the .duration field of each inserted boundary event.
    boundevents = r[:, 0].astype(float) - 1.0
    if len(base_durations) > 1:
        cums = np.concatenate([[0.0], np.cumsum(base_durations[:-1].astype(float))])
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

    # Insert boundary events (always, even if no original events)
    if True:
        bound_type = "boundary"
        for i in range(len(boundevents)):
            be = float(boundevents[i])
            if be > 0 and be < (newn + 1):
                events_out.append(
                    {
                        "type": bound_type,
                        "latency": be,
                        "duration": float(
                            durations[i]
                            if i < len(durations)
                            else (base_durations[i] if i < len(base_durations) else 0.0)
                        ),
                    }
                )

    # Remove nested boundary events that were absorbed into new boundaries.
    # These are pre-existing boundaries that fell inside removal regions;
    # their durations were added to the new boundary's .duration field above.
    # Match by original latency (adjusted by prior region removals).
    if nested_to_remove:
        # Collect adjusted latencies of nested boundaries to remove
        nested_lats = set()
        for idx in nested_to_remove:
            if idx < len(ori_events):
                # The latency was already adjusted in events_out during the
                # latency shift loop; find the event by identity (same dict ref
                # won't work since we copied). Use a latency-based search on
                # the boundary events that are NOT newly inserted.
                nested_lats.add(float(ori_events[idx].get("latency", float("nan"))))
        if nested_lats:
            # Build set of adjusted latencies for nested events
            # ori_lat was captured before adjustments; recompute adjusted lat
            ori_lat_arr = np.array([float(ev.get("latency", float("nan"))) for ev in ori_events], dtype=float)
            adj_lat = ori_lat_arr.copy()
            for beg, end in r:
                span = int(end - beg + 1)
                adj_lat[ori_lat_arr > beg] -= span
            adj_nested_lats = set(float(adj_lat[i]) for i in nested_to_remove if i < len(adj_lat))

            cleaned = []
            for ev in events_out:
                if _is_boundary_event(ev) and float(ev.get("latency", -1)) in adj_nested_lats:
                    adj_nested_lats.discard(float(ev.get("latency", -1)))
                    continue  # skip this nested boundary
                cleaned.append(ev)
            events_out = cleaned

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
            if (
                merged_events
                and _is_boundary_event(ev)
                and _is_boundary_event(merged_events[-1])
                and np.isclose(float(ev.get("latency", 0.0)), float(merged_events[-1].get("latency", 0.0)))
            ):
                prev_dur = float(merged_events[-1].get("duration", 0.0) or 0.0)
                cur_dur = float(ev.get("duration", 0.0) or 0.0)
                merged_events[-1]["duration"] = prev_dur + cur_dur
            else:
                merged_events.append(ev)
        events_out = merged_events

    return newx, newt, events_out, boundevents


def eeg_eegrej(EEG, regions):
    """Reject EEG data segments specified by regions.

    Parameters
    ----------
    EEG : dict
        EEG data structure
    regions : array-like
        Regions to reject, shape (n_regions, 2) or (n_regions, 4)

    Returns
    -------
    EEG : dict
        Updated EEG data structure with rejected segments removed
    """
    EEG = deepcopy(EEG)
    if regions is None or len(regions) == 0:
        return EEG

    # Round first like MATLAB, then convert to int
    regions = np.asarray(regions, dtype=float)
    regions = round_mat(regions).astype(np.int64)

    # sort rows like MATLAB
    if regions.shape[1] > 2:
        regions = regions[np.argsort(regions[:, 2])]
    else:
        regions = regions[np.argsort(regions[:, 0])]

    # handle eegplot-style regions [.. .. beg end]
    if regions.shape[1] > 2:
        regions = regions[:, 2:4]

    regions = _combine_regions(regions)

    # Use original events; backend will handle pruning, shifting, and boundary insertion
    events = list(EEG.get("event", []))

    # call _eegrej backend
    xdur = float(EEG["xmax"] - EEG["xmin"])
    data_out, xmax_rel, event2, boundevents = _eegrej(EEG["data"], regions, xdur, events)

    # finalize core fields
    EEG["data"] = data_out
    EEG["pnts"] = int(data_out.shape[1])
    EEG["xmax"] = float(EEG["xmin"] + xmax_rel)
    EEG['times'] = np.linspace(EEG['xmin'] * 1000, EEG['xmax'] * 1000, EEG['pnts'], dtype=float)

    # Use backend-generated events list (boundary insertion already done by _eegrej)
    EEG["event"] = list(event2) if isinstance(event2, list) else []
    EEG["event"].sort(key=lambda e: e.get("latency", float("inf")))

    if len(EEG["event"]) > 1 and EEG["event"][-1].get("latency", 0) - 0.5 > EEG["pnts"] and EEG.get("trials", 1) == 1:
        EEG["event"].pop()

    # light duplicate cleanup mirroring MATLAB edge cases
    if len(EEG["event"]) > 1 and EEG["event"][0].get("latency") == 0:
        EEG["event"] = EEG["event"][1:]
    if len(EEG["event"]) > 1 and EEG["event"][-1].get("latency") == EEG["pnts"]:
        EEG["event"] = EEG["event"][:-1]
    if len(EEG["event"]) > 2:
        if EEG["event"][-1].get("latency") == EEG["event"][-2].get("latency"):
            if EEG["event"][-1].get("type") == EEG["event"][-2].get("type"):
                EEG["event"].pop()

    # make sure that each newly inserted boundary event has all fields
    extra_fields = set().union(*(ev.keys() for ev in EEG['event'])) - {"type", "latency", "duration"}
    for ev in EEG['event']:
        if _is_boundary_event(ev):
            if extra_fields - set(ev.keys()):
                for f in extra_fields:
                    if f not in ev:
                        ev[f] = np.array([])

    return EEG


def _combine_regions(regs):
    if len(regs) == 0:
        return regs
    regs = np.array(sorted(regs.tolist(), key=lambda r: (r[0], r[1])), dtype=np.int64)
    merged = [regs[0].tolist()]
    for beg, end in regs[1:]:
        mbeg, mend = merged[-1]
        if beg <= mend + 1:
            merged[-1][1] = max(mend, end)
        else:
            merged.append([beg, end])
    newregs = np.asarray(merged, dtype=np.int64)
    if newregs.shape[0] != regs.shape[0]:
        print("Warning: overlapping regions detected and fixed in eeg_eegrej")
    return newregs


def _find_boundary_event_indices(events):
    idx = []
    for i, ev in enumerate(events):
        t = ev.get("type")
        if isinstance(t, str) and t.lower() == "boundary":
            idx.append(i)
        elif isinstance(t, (int, float)) and int(t) == -99:
            idx.append(i)
    return np.array(idx, dtype=int)


def _insert_boundaries(events, old_pnts, regions):
    # Build kept segments in 1-based indices
    kept = []
    cursor = 1
    for beg, end in regions:
        if cursor <= beg - 1:
            kept.append([cursor, beg - 1])
        cursor = end + 1
    if cursor <= old_pnts:
        kept.append([cursor, old_pnts])

    out = [dict(ev) for ev in events]
    run_len = 0
    for i in range(len(kept) - 1):
        seg_len = kept[i][1] - kept[i][0] + 1
        run_len += seg_len
        rem_beg, rem_end = regions[i]
        rem_len = int(rem_end - rem_beg + 1)
        out.append(
            {
                "type": "boundary",
                "latency": float(run_len + 1),
                "duration": float(rem_len),
            }
        )
    return out
