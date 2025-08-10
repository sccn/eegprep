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

    # remove events that fall within any region, except boundary events
    events = list(EEG.get("event", []))
    if events:
        ev_lats = np.array([float(e["latency"]) for e in events])
        kill = np.zeros(len(events), dtype=bool)
        for beg, end in regions:
            kill |= (ev_lats >= beg) & (ev_lats <= end)
        bidx = _find_boundary_event_indices(events)
        kill[bidx] = False
        events = [ev for i, ev in enumerate(events) if not kill[i]]

    # call eegrej backend
    xdur = float(EEG["xmax"] - EEG["xmin"])
    data_out, xmax_rel, event2, boundevents = eegrej(EEG["data"], regions, xdur, events)

    # finalize core fields
    old_pnts = int(EEG["pnts"])
    EEG["data"] = data_out
    EEG["pnts"] = int(data_out.shape[1])
    EEG["xmax"] = float(EEG["xmax"] + EEG["xmin"])

    # insert boundary events into our pruned events, then consistency trims
    EEG["event"] = _insert_boundaries(events, old_pnts, regions)
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