"""EEG data structure comparison utilities.

This module provides functions for comparing EEG data structures and reporting
differences between them.
"""

import sys
import math
from collections.abc import Sequence
import numpy as np

def eeg_compare(eeg1, eeg2, verbose_level=0, trigger_error=False):
    """Compare two EEG-like structures, reporting differences to stderr.

    Parameters
    ----------
    eeg1 : dict or object
        First EEG structure to compare.
    eeg2 : dict or object
        Second EEG structure to compare.
    verbose_level : int, optional
        Level of verbosity for output. Default 0.
    trigger_error : bool, optional
        Whether to raise an error if differences are found. Default False.

    Returns
    -------
    bool
        True if comparison completed (differences may still exist).
    """
    
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
                return bool(np.array_equal(np.array(a), np.array(b), equal_nan=True))
            except:
                pass
        # Handle numpy arrays in general comparison
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            try:
                return bool(np.array_equal(a, b, equal_nan=True))
            except:
                pass
        # Handle scalar vs array comparisons
        if isinstance(a, np.ndarray) and np.isscalar(b):
            try:
                return bool(np.all(a == b))
            except:
                pass
        if isinstance(b, np.ndarray) and np.isscalar(a):
            try:
                return bool(np.all(b == a))
            except:
                pass
        # Final comparison - ensure we return a boolean
        try:
            result = a == b
            if isinstance(result, np.ndarray):
                return bool(result.all())
            return bool(result)
        except:
            return False
    
    print('\nField analysis: (no entries means OK)')
    
    # Collect differences for error reporting
    differences = []
    # Handle both dictionary-like objects and objects with __dict__
    if hasattr(eeg1, 'keys'):
        # Dictionary-like object
        fields1 = eeg1.keys()
        get_val1 = lambda f: eeg1.get(f, None)
        # Handle case where eeg2 might be a numpy array or other non-dict object
        if hasattr(eeg2, 'keys'):
            has_field2 = lambda f: f in eeg2
            get_val2 = lambda f: eeg2.get(f, None)
        else:
            has_field2 = lambda f: False  # No fields available
            get_val2 = lambda f: None
    else:
        # Object with __dict__
        fields1 = getattr(eeg1, '__dict__', {}).keys()
        get_val1 = lambda f: getattr(eeg1, f, None)
        has_field2 = lambda f: hasattr(eeg2, f)
        get_val2 = lambda f: getattr(eeg2, f, None)
    
    for field in fields1:
        if not has_field2(field):
            error_msg = f'Field {field} missing in second dataset'
            print(f'    {error_msg}', file=sys.stderr)
            differences.append(error_msg)
        else:
            v1 = get_val1(field)
            v2 = get_val2(field)
            if not isequaln(v1, v2):
                name = field.lower()
                if any(sub in name for sub in ('filename', 'datfile')):
                    print(f'    Field {field} differs (ok, supposed to differ)')
                elif any(sub in name for sub in ('subject', 'session', 'run', 'task')):
                    error_msg = f'Field {field} differs ("{v1}" vs "{v2}")'
                    print(f'    {error_msg}', file=sys.stderr)
                    differences.append(error_msg)
                elif any(sub in name for sub in ('eventdescription')):
                    n1 = len(v1) if isinstance(v1, Sequence) else 1
                    n2 = len(v2) if isinstance(v2, Sequence) else 1
                    error_msg = f'Field {field} differs (n={n1} vs n={n2})'
                    print(f'    {error_msg}', file=sys.stderr)
                    differences.append(error_msg)
                elif any(sub in name for sub in ('chanlocs', 'event', 'reject')):
                    pass
                    # For complex nested structures, provide more detailed info
                else:
                    error_msg = f'Field {field} differs'
                    print(f'    {error_msg}', file=sys.stderr)
                    differences.append(error_msg)
    # compare xmin/xmax
    for attr in ('xmin', 'xmax'):
        x1 = get_val1(attr)
        x2 = get_val2(attr)
        if not isequaln(x1, x2):
            diff = (x1 or 0) - (x2 or 0)
            error_msg = f'Difference between {attr} is {diff:1.6f} sec'
            print(f'    {error_msg}', file=sys.stderr)
            differences.append(error_msg)

    # channel locations
    print('Chanlocs analysis:')
    chans1 = get_val1('chanlocs')
    if chans1 is None or (isinstance(chans1, np.ndarray) and len(chans1) == 0):
        chans1 = []
    chans2 = get_val2('chanlocs')
    if chans2 is None or (isinstance(chans2, np.ndarray) and len(chans2) == 0):
        chans2 = []
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
                if verbose_level > 0:
                    print(f'    Channel {c1["labels"]} differs from {c2["labels"]}', file=sys.stderr)
        if coord_diff:
            error_msg = f'{coord_diff} channel coordinates differ'
            print(f'    {error_msg}', file=sys.stderr)
            differences.append(error_msg)
        else:
            print('    All channel coordinates are OK')
        if label_diff:
            error_msg = f'{label_diff} channel label(s) differ'
            print(f'    {error_msg}', file=sys.stderr)
            differences.append(error_msg)
        else:
            print('    All channel labels are OK')
    else:
        error_msg = 'Different numbers of channels'
        print(f'    {error_msg}', file=sys.stderr)
        differences.append(error_msg)

    # events
    print('Event analysis:')
    ev1 = get_val1('event')
    if ev1 is None or (isinstance(ev1, np.ndarray) and len(ev1) == 0):
        ev1 = []
    ev2 = get_val2('event')
    if ev2 is None or (isinstance(ev2, np.ndarray) and len(ev2) == 0):
        ev2 = []
    if len(ev1) != len(ev2):
        error_msg = f'Different numbers of events {len(ev1)} vs {len(ev2)}'
        print(f'    {error_msg}', file=sys.stderr)
        differences.append(error_msg)
        # print the first event of each
        if verbose_level > 0:
            if len(ev1) > 0:
                print(f'    First event of first dataset: {ev1[0]}', file=sys.stderr)
            if len(ev2) > 0:
                print(f'    First event of second dataset: {ev2[0]}', file=sys.stderr)
    else:
        if len(ev1) == 0:
            print('    All events OK (empty)')
        else:
            f1 = set(ev1[0].keys())
            f2 = set(ev2[0].keys())
            if f1 != f2:
                error_msg = 'Not the same number of event fields'
                print(f'    {error_msg}', file=sys.stderr)
                differences.append(error_msg)
            for fld in f1:
                diffs = []
                if fld.lower() == 'latency':
                    diffs = [e1['latency'] - e2['latency'] for e1, e2 in zip(ev1, ev2)]
                    nonzero = [d for d in diffs if d != 0]
                    if nonzero:
                        pct = len(nonzero) / len(diffs) * 100
                        avg = sum(abs(d) for d in nonzero) / len(nonzero)
                        error_msg = f'Event latency ({pct:2.1f} %) not OK (abs diff {avg:1.4f} samples)'
                        print(f'    {error_msg}', file=sys.stderr)
                        differences.append(error_msg)
                        # print('    ******** (see plot)')
                        # import matplotlib.pyplot as plt
                        # plt.plot(diffs)
                        # plt.show()
                else:
                    diffs = [not isequaln(e1.get(fld, None), e2.get(fld, None)) for e1, e2 in zip(ev1, ev2)]
                    if any(diffs):
                        pct = sum(diffs) / len(diffs) * 100
                        error_msg = f'Event fields "{fld}" are NOT OK ({pct:2.1f} % of them)'
                        print(f'    {error_msg}', file=sys.stderr)
                        differences.append(error_msg)
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

    # Raise error if differences found and trigger_error is True
    if trigger_error and differences:
        error_message = f"EEG comparison failed with {len(differences)} differences:\n" + "\n".join(f"  - {diff}" for diff in differences)
        raise ValueError(error_message)
    
    return True

# add test data and compare with it

# load test data
if __name__ == '__main__':
    from eegprep import pop_loadset
    eeg1 = pop_loadset('../../data/eeglab_data_tmp.set')
    eeg2 = pop_loadset('../../data/eeglab_data_tmp.set')

    # compare
    eeg_compare(eeg1, eeg2)