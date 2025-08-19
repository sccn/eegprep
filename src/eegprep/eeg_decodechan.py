def eeg_decodechan(
    chanlocs,
    chanstr,
    field="labels",
    ignoremissing=False,
):
    """
    Resolve channel identifiers to 1-based indices and labels.

    Supports:
      - chanlocs as a list-like of dicts, or a dict with key "chanlocs".
      - chanstr as an iterable of strings and/or integers.
      - Matching on the specified field (e.g., "labels" or "type").
      - Numeric 1-based indices as input (returned directly after validation).
      - Empty chanlocs with purely numeric input (indices passthrough).

    Returns:
      (chaninds, chanlist_out)
        chaninds: sorted list of 1-based indices
        chanlist_out: list of labels/types from chanlocs for those indices
                      or the indices themselves if chanlocs is empty
    """

    # Unwrap {"chanlocs": [...]}
    if isinstance(chanlocs, dict) and "chanlocs" in chanlocs:
        chanlocs = chanlocs["chanlocs"]

    # Basic indexability check
    if not hasattr(chanlocs, "__len__") or not hasattr(chanlocs, "__getitem__"):
        raise TypeError("chanlocs must be an indexable sequence of dictionaries or {'chanlocs': [...]}")

    nchan = len(chanlocs)

    # Normalize chanstr into a flat Python list
    try:
        seq = list(chanstr)
    except Exception as e:
        raise TypeError("chanstr must be an iterable of strings/integers") from e

    # Detect numeric-only request (ints or strings that are pure integers)
    numeric_req = []
    nonnum_req = []
    for x in seq:
        if isinstance(x, (int, float)) and float(x).is_integer():
            numeric_req.append(int(x))
        else:
            xs = str(x).strip()
            if xs.isdigit():
                numeric_req.append(int(xs))
            else:
                nonnum_req.append(xs)

    # Case 1: purely numeric input → treat as 1-based indices
    if len(nonnum_req) == 0:
        chaninds = sorted(int(i) for i in numeric_req)
        if nchan == 0:
            # passthrough when chanlocs is empty
            return chaninds, chaninds[:]
        if any(i <= 0 or i > nchan for i in chaninds):
            raise ValueError("Channel index out of range")
        chanlist_out = [str(chanlocs[i - 1][field]) for i in chaninds]
        return chaninds, chanlist_out

    # Case 2: name/type matching (optionally mixed with numeric indices)
    # Start with validated numeric indices (if any)
    chaninds = []
    if numeric_req:
        if nchan == 0:
            # cannot validate numbers without chanlocs
            raise ValueError("Numeric indices cannot be validated because chanlocs is empty")
        if any(i <= 0 or i > nchan for i in numeric_req):
            raise ValueError("Channel index out of range")
        chaninds.extend(int(i) for i in numeric_req)

    # Prepare lowercase lookup for the chosen field
    if nchan == 0:
        # Nothing to match against
        if ignoremissing:
            chaninds = sorted(set(chaninds))
            return chaninds, chaninds[:]
        raise ValueError("chanlocs is empty; cannot resolve non-numeric channel names")

    try:
        alllabs = [str(c[field]).strip().lower() for c in chanlocs]
    except Exception as e:
        raise ValueError(f"Field '{field}' not found in chanlocs dictionaries") from e

    # Match each requested name (case-insensitive), keep duplicates if any
    for name in (s.lower().strip() for s in nonnum_req):
        matches = [i + 1 for i, lab in enumerate(alllabs) if lab == name]
        if matches:
            chaninds.extend(matches)
        elif not ignoremissing:
            raise ValueError(f"Channel '{name}' not found")

    # Final validations and outputs
    if any(i <= 0 or i > nchan for i in chaninds):
        raise ValueError("Channel index out of range")

    chaninds = sorted(chaninds)
    chanlist_out = [str(chanlocs[i - 1][field]) for i in chaninds]
    return chaninds, chanlist_out