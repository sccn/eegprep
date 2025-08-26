import numpy as np
from copy import deepcopy
import numpy as np
from typing import List, Dict, Optional, Tuple

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

def _eegrej(indata, regions, timelength, events: Optional[List[Dict]] = None) -> Tuple[np.ndarray, float, List[Dict], np.ndarray]:
    """
    Remove [beg end] sample ranges (1-based, inclusive) from continuous data
    and update events (list of dictionaries) in the MATLAB EEGLAB style.

    Inputs
      - indata: 2D array shaped (channels, frames)
      - regions: array-like with shape (n_regions, 2), 1-based [beg end] per row
      - timelength: total duration of the original data in seconds
      - events: list of dicts with at least key 'latency'; optional keys include
                'type' and 'duration'. If None or empty, boundary events will
                still be inserted based on regions.

    Returns
      - outdata: data with columns removed
      - newt: new total time in seconds
      - events_out: updated events list of dictionaries (with inserted boundaries)
      - boundevents: boundary latencies (float, 1-based, with +0.5 convention)
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
        rounded = np.round(boundevents, 12)
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


def eeg_eegrej(EEG, regions):
    EEG = deepcopy(EEG)
    if regions is None or len(regions) == 0:
        return EEG

    # Round first like MATLAB, then convert to int
    regions = np.asarray(regions, dtype=float)
    regions = np.round(regions).astype(np.int64)
    
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
    old_pnts = int(EEG["pnts"])
    EEG["data"] = data_out
    EEG["pnts"] = int(data_out.shape[1])
    EEG["xmax"] = float(EEG["xmin"] + xmax_rel)

    # Use backend-generated events list and sort
    EEG["event"] = list(event2) if isinstance(event2, list) else []
    EEG["event"].sort(key=lambda e: e.get("latency", float("inf")))

    # Ensure a boundary is present at each kept/run boundary with integer latency and correct duration
    # This mirrors historical behavior expected by tests (new boundary at run_len+1 with duration = removed length)
    def _ensure_integer_boundaries(ev_list, old_pnts, regs):
        kept = []
        cursor = 1
        for beg, end in regs:
            if cursor <= beg - 1:
                kept.append([cursor, beg - 1])
            cursor = end + 1
        if cursor <= old_pnts:
            kept.append([cursor, old_pnts])

        out = list(ev_list)
        run_len = 0
        for i in range(len(kept) - 1):
            seg_len = kept[i][1] - kept[i][0] + 1
            run_len += seg_len
            rem_beg, rem_end = regs[i]
            rem_len = int(rem_end - rem_beg + 1)
            new_lat = float(run_len + 1)
            # find existing boundary at this integer latency
            found = False
            for ev in out:
                if ev.get("type") == "boundary" and float(ev.get("latency", -1.0)) == new_lat:
                    # update duration to removed length
                    ev["duration"] = float(rem_len)
                    found = True
                    break
            if not found:
                out.append({
                    "type": "boundary",
                    "latency": new_lat,
                    "duration": float(rem_len),
                })
        return out

    EEG["event"] = _ensure_integer_boundaries(EEG["event"], old_pnts, regions)
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
        out.append({
            "type": "boundary",
            "latency": float(run_len + 1),
            "duration": float(rem_len),
        })
    return out