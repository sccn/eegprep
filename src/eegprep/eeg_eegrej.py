import numpy as np
from copy import deepcopy
from eegprep.eegrej import eegrej  # expects: eegrej(data, regions, xdur, events) -> (data_out, xmax_rel, event2, boundevents)

def eeg_eegrej(EEG, regions):
    EEG = deepcopy(EEG)
    if regions is None or len(regions) == 0:
        return EEG

    regions = np.asarray(regions, dtype=np.int64)
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

    # call eegrej backend
    xdur = float(EEG["xmax"] - EEG["xmin"])
    data_out, xmax_rel, event2, boundevents = eegrej(EEG["data"], regions, xdur, events)

    # finalize core fields
    old_pnts = int(EEG["pnts"])
    EEG["data"] = data_out
    EEG["pnts"] = int(data_out.shape[1])
    EEG["xmax"] = float(EEG["xmax"] + EEG["xmin"])

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