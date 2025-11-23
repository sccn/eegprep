"""EEG object wrapper for dict-based datasets."""

import copy
import os
import importlib

from eegprep.pop_loadset import pop_loadset
from eegprep.pop_select import pop_select  # ensure availability via globals


class EEGobj:
    """Wrapper class for EEG datasets stored as dictionaries.

    Provides attribute access to EEG fields and method calls to eegprep
    functions.
    """

    def __init__(self, EEG_or_path):
        """Initialize from an EEG dict or a file path string.

        - If string: loads dataset with pop_loadset(path).
        - If dict: uses it directly.
        """

    # Internal helper to resolve and call an eegprep function name
    def _call_eegprep(self, fname, *args, **kwargs):
        if isinstance(EEG_or_path, str):
            EEG = pop_loadset(EEG_or_path)
        elif isinstance(EEG_or_path, dict):
            EEG = EEG_or_path
        else:
            raise TypeError("EEGobj requires a dict or a file path string")
        object.__setattr__(self, 'EEG', EEG)

    # Internal helper to resolve and call an eegprep function name
    def _call_eegprep(self, fname, *args, **kwargs):
        import types
        def _resolve(n):
            # Try globals first (for imported functions)
            cand = globals().get(n)
            if callable(cand):
                return cand
            # Try lazy import from eegprep package
            try:
                mod = importlib.import_module(f"eegprep.{n}")
                sub = getattr(mod, n, None)
                if callable(sub):
                    return sub
            except Exception:
                pass
            # Try as submodule of eegprep
            try:
                import eegprep as eegpkg
                cand = getattr(eegpkg, n, None)
                if callable(cand):
                    return cand
                if isinstance(cand, types.ModuleType):
                    sub = getattr(cand, n, None)
                    if callable(sub):
                        return sub
            except Exception:
                pass
            return None

        func = _resolve(fname)
        if func is None:
            raise AttributeError(fname)

        # MATLAB-style key/value pairs support
        args2 = args
        kwargs2 = dict(kwargs)
        if len(args) >= 2 and isinstance(args[0], (str, bytes)):
            if len(args) % 2 != 0:
                raise ValueError("Key/value arguments must be in pairs")
            kv = {}
            for i in range(0, len(args), 2):
                key = args[i]
                val = args[i + 1]
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                if not isinstance(key, str):
                    raise ValueError("Keys must be strings")
                kv[key] = val
            args2 = ()
            kwargs2 = {**kv, **kwargs2}

        # normalize common plural keys
        key_map = {'trials': 'trial', 'channels': 'channel', 'points': 'point', 'times': 'time'}
        kwargs2 = {key_map.get(k, k): v for k, v in kwargs2.items()}

        new_eeg = copy.deepcopy(self.EEG)
        result = func(new_eeg, *args2, **kwargs2)
        if isinstance(result, dict):
            object.__setattr__(self, 'EEG', result)
        elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], dict):
            object.__setattr__(self, 'EEG', result[0])
        else:
            object.__setattr__(self, 'EEG', new_eeg)
        return self.EEG

    def __getattr__(self, name):
        """Access EEG fields or eegprep functions.

        - If 'name' is a key in EEG, return EEG[name] (convenience).
        - If 'name' is a function in eegprep, return a wrapper that:
          self.EEG = func(deepcopy(self.EEG), ...)
          and returns updated EEG for convenience.
        """
        eeg = object.__getattribute__(self, 'EEG')
        if isinstance(eeg, dict) and name in eeg:
            return eeg[name]

        def wrapper(*args, **kwargs):
            return self._call_eegprep(name, *args, **kwargs)
        return wrapper

    def __setattr__(self, name, value):
        """Set attributes on the underlying EEG dict when possible, else on the
        wrapper."""
        if name == 'EEG':
            object.__setattr__(self, name, value)
            return
        eeg = object.__getattribute__(self, 'EEG')
        if isinstance(eeg, dict):
            eeg[name] = value
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        """Multi-line, MNE-like summary of the EEG object.

        Shows key metadata, data shape, sampling info, time span, and brief
        events/channels info.
        """
        eeg = self.EEG
        if not isinstance(eeg, dict):
            return f"<EEGobj: {type(eeg)}>"

        def _safe(val, default=''):
            try:
                # Avoid numpy truth-value ambiguity
                if isinstance(val, (list, tuple, dict)):
                    return val
                if hasattr(val, 'shape'):
                    return val
                return val if val is not None else default
            except Exception:
                return default

        setname = _safe(eeg.get('setname'), '') or ''
        fname = _safe(eeg.get('filename'), '') or ''
        fpath = _safe(eeg.get('filepath'), '') or ''
        nbchan = eeg.get('nbchan')
        pnts = eeg.get('pnts')
        trials = eeg.get('trials')
        srate = eeg.get('srate')
        xmin = eeg.get('xmin')
        xmax = eeg.get('xmax')
        data = eeg.get('data')
        ev = eeg.get('event')
        if ev is None:
            ev = []
        clocs = eeg.get('chanlocs')
        if clocs is None:
            clocs = []

        # Data shape and duration
        if data is not None and hasattr(data, 'shape'):
            shape_str = ' x '.join(str(d) for d in data.shape)
        else:
            shape_str = 'unknown'

        try:
            duration = None
            if xmin is not None and xmax is not None:
                duration = float(xmax) - float(xmin)
            elif pnts is not None and srate is not None:
                duration = (float(pnts) - 1.0) / float(srate)
        except Exception:
            duration = None

        # Events summary (up to top-3 types)
        try:
            evcnt = len(ev) if isinstance(ev, (list, tuple)) else (int(ev.size) if hasattr(ev, 'size') else 0)
        except Exception:
            evcnt = 0
        ev_types = {}
        try:
            iterable_ev = ev
            if hasattr(ev, 'tolist'):
                iterable_ev = ev.tolist()
            if isinstance(iterable_ev, (list, tuple)):
                for e in iterable_ev:
                    if isinstance(e, dict) and 'type' in e:
                        t = e['type']
                        if isinstance(t, bytes):
                            try:
                                t = t.decode('utf-8')
                            except Exception:
                                pass
                        ev_types[t] = ev_types.get(t, 0) + 1
        except Exception:
            pass
        top_ev = ', '.join(f"{k}:{v}" for k, v in list(ev_types.items())[:3]) if ev_types else '—'

        # Channel type summary (if available)
        ch_types = {}
        try:
            iterable_cl = clocs
            if hasattr(clocs, 'tolist'):
                iterable_cl = clocs.tolist()
            for ch in (iterable_cl if isinstance(iterable_cl, (list, tuple)) else []):
                if isinstance(ch, dict) and 'type' in ch:
                    t = ch['type']
                    if isinstance(t, bytes):
                        try:
                            t = t.decode('utf-8')
                        except Exception:
                            pass
                    ch_types[t] = ch_types.get(t, 0) + 1
        except Exception:
            pass
        ch_types_str = ', '.join(f"{k}:{v}" for k, v in ch_types.items()) if ch_types else '—'

        header = setname or fname or '<unnamed EEG>'
        fileline = os.path.join(fpath, fname) if (fpath and fname) else (fname or fpath or '—')

        lines = []
        lines.append(f"EEG | {header}")
        lines.append(f"  Data shape      : {shape_str}")
        lines.append(f"  Channels        : {nbchan}")
        lines.append(f"  Sampling freq.  : {srate} Hz")
        if trials is not None:
            lines.append(f"  Trials          : {trials}")
        if xmin is not None and xmax is not None:
            lines.append(f"  Time            : [{xmin}, {xmax}] s")
        if duration is not None:
            lines.append(f"  Duration        : {duration:.3f} s")
        lines.append(f"  Events          : {evcnt} (types: {top_ev})")
        lines.append(f"  Channel types   : {ch_types_str}")
        lines.append(f"  File            : {fileline}")

        return '\n'.join(lines)

    __str__ = __repr__


if __name__ == '__main__':
    obj = EEGobj('data/eeglab_data.set')
    print(obj)