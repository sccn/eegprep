import sys
import math
from collections.abc import Sequence
import numpy as np

def eeg_compare(eeg1, eeg2):
    
    def isequaln(a, b):
        """Treat None and NaN as equal, otherwise compare by value."""
        # both None
        if a is None and b is None:
            return True
        # None vs NaN
        if a is None and isinstance(b, float) and math.isnan(b):
            return True
        if b is None and isinstance(a, float) and math.isnan(a):
            return True
        # both NaN
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        # arrays with NaN
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            try:
                return np.array_equal(np.array(a), np.array(b), equal_nan=True)
            except:
                pass
        return a == b
    
    """Compare two EEG-like structures, reporting differences to stderr."""
    print('Field analysis:')
    for field in getattr(eeg1, '__dict__', eeg1).keys():
        if not hasattr(eeg2, field):
            print(f'    Field {field} missing in second dataset', file=sys.stderr)
        else:
            v1 = getattr(eeg1, field, None)
            v2 = getattr(eeg2, field, None)
            if not isequaln(v1, v2):
                name = field.lower()
                if any(sub in name for sub in ('filename', 'datfile')):
                    print(f'    Field {field} differs (ok, supposed to differ)')
                elif any(sub in name for sub in ('subject', 'session', 'run', 'task')):
                    print(f'    Field {field} differs ("{v1}" vs "{v2}")', file=sys.stderr)
                elif any(sub in name for sub in ('eventdescription', 'event')):
                    n1 = len(v1) if isinstance(v1, Sequence) else 1
                    n2 = len(v2) if isinstance(v2, Sequence) else 1
                    print(f'    Field {field} differs (n={n1} vs n={n2})', file=sys.stderr)
                else:
                    print(f'    Field {field} differs', file=sys.stderr)
    # compare xmin/xmax
    for attr in ('xmin', 'xmax'):
        x1 = getattr(eeg1, attr, None)
        x2 = getattr(eeg2, attr, None)
        if not isequaln(x1, x2):
            diff = (x1 or 0) - (x2 or 0)
            print(f'    Difference between {attr} is {diff:1.6f} sec', file=sys.stderr)

    # channel locations
    print('Chanlocs analysis:')
    chans1 = eeg1['chanlocs'] # need to fuse with chaninfo   
    chans2 = eeg2['chanlocs'] # need to fuse with chaninfo
    if len(chans1) == len(chans2):
        coord_diff = label_diff = 0
        for c1, c2 in zip(chans1, chans2):
            c1_xyz = (c1['X'], c1['Y'], c1['Z'])
            c2_xyz = (c2['X'], c2['Y'], c2['Z'])
            if (any(v is None for v in c1_xyz) and not any(v is None for v in c2_xyz)) \
               or (any(v is None for v in c2_xyz) and not any(v is None for v in c1_xyz)) \
               or (all(v is not None for v in (*c1_xyz,)) and 
                   sum(abs(a - b) for a, b in zip(c1_xyz, c2_xyz)) > 1e-12):
                coord_diff += 1
            if c1['labels'] != c2['labels']:
                label_diff += 1
        if coord_diff:
            print(f'    {coord_diff} channel coordinates differ', file=sys.stderr)
        else:
            print('    All channel coordinates are OK')
        if label_diff:
            print(f'    {label_diff} channel label(s) differ', file=sys.stderr)
        else:
            print('    All channel labels are OK')
    else:
        print('    Different numbers of channels', file=sys.stderr)

    # events
    print('Event analysis:')
    ev1, ev2 = eeg1['event'], eeg2['event']
    if len(ev1) != len(ev2):
        print('    Different numbers of events', file=sys.stderr)
    elif not ev1:
        print('    All events OK (empty)')
    else:
        f1 = set(ev1[0].keys())
        f2 = set(ev2[0].keys())
        if f1 != f2:
            print('    Not the same number of event fields', file=sys.stderr)
        for fld in f1:
            diffs = []
            if fld.lower() == 'latency':
                diffs = [e1['latency'] - e2['latency'] for e1, e2 in zip(ev1, ev2)]
                nonzero = [d for d in diffs if d != 0]
                if nonzero:
                    pct = len(nonzero) / len(diffs) * 100
                    avg = sum(abs(d) for d in nonzero) / len(nonzero)
                    print(f'    Event latency ({pct:2.1f} %) not OK (abs diff {avg:1.4f} samples)', file=sys.stderr)
                    # print('    ******** (see plot)')
                    # import matplotlib.pyplot as plt
                    # plt.plot(diffs)
                    # plt.show()
            else:
                diffs = [not isequaln(getattr(e1, fld, None), getattr(e2, fld, None)) for e1, e2 in zip(ev1, ev2)]
                if any(diffs):
                    pct = sum(diffs) / len(diffs) * 100
                    print(f'    Event fields "{fld}" are NOT OK ({pct:2.1f} % of them)', file=sys.stderr)
        print('    All other events OK')

    # epochs
    # if 'epoch' in eeg1:
    #     print('Epoch analysis:')
    #     ep1, ep2 =  eeg1['epoch'], eeg2['epoch']
    #     if len(ep1) != len(ep2):
    #         print('    Different numbers of epochs', file=sys.stderr)
    #     else:
    #         fields = ep1[0].keys()
    #         all_ok = True
    #         for fld in fields:
    #             diffs = [not isequaln(getattr(e1, fld, None), getattr(e2, fld, None)) for e1, e2 in zip(ep1, ep2)]
    #             if any(diffs):
    #                 pct = sum(diffs) / len(diffs) * 100
    #                 print(f'    Epoch fields "{fld}" are NOT OK ({pct:2.1f} % of them)', file=sys.stderr)
    #                 all_ok = False
    #         if all_ok:
    #             print('    All epoch and all epoch fields are OK')

    return True