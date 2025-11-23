"""EEG epoching utilities."""

import numpy as np
import copy
import re
from eegprep.epoch import epoch
from eegprep.eeg_findboundaries import eeg_findboundaries
from eegprep.pop_select import pop_select
from eegprep.eeg_checkset import eeg_checkset


def pop_epoch(EEG, types=None, lim=None, **kwargs):
    """Convert a continuous EEG dataset to epoched data by extracting data
    epochs time locked to specified event types or event indices.

    May also sub-epoch an already epoched dataset.

    Python translation of EEGLAB's pop_epoch function.

    Usage:
        EEG_out = pop_epoch(EEG)  # GUI mode not supported
        EEG_out = pop_epoch(EEG, types, timelimits)
        EEG_out, indices = pop_epoch(EEG, types, timelimits, **kwargs)

    Inputs:
        EEG        - Input EEG dataset (dict). Data may already be epoched.
        types      - String (regular expression) or list of event types to time
                     lock to. Default is [] which means to extract epochs
                     locked to every single event.
        lim        - Epoch latency limits [start end] in seconds relative to
                     the time-locking event. Default: [-1, 2]

    Optional keyword arguments:
        eventindices - List of event indices to use for epoching (0-based)
        timeunit     - 'seconds' or 'points'. Default: 'points'
        newname      - Name for the new dataset
        valuelim     - Rejection limits [min max]. Default: [-Inf, Inf]
        epochinfo    - 'yes' or 'no'. Default: 'yes'

    Outputs:
        EEG_out    - Output epoched dataset
        indices    - Indices of accepted epochs (0-based)

    Note: This function calls the epoch() function to do the actual epoching.
    """
    # Input validation
    if EEG is None:
        raise ValueError('pop_epoch: EEG dataset is required')

    # Handle multiple datasets (not implemented)
    if isinstance(EEG, list) and len(EEG) > 1:
        raise NotImplementedError('pop_epoch: multiple datasets not supported')

    if isinstance(EEG, list):
        EEG = EEG[0]

    # Check for empty event structure
    if 'event' not in EEG or EEG['event'] is None or len(EEG['event']) == 0:
        if EEG.get('trials', 1) > 1 and EEG.get('xmin', 0) <= 0 and EEG.get('xmax', 0) >= 0:
            print("No EEG.event structure found: creating events of type 'TLE' (Time-Locking Event) at time 0")
            # Create TLE events
            EEG['event'] = []
            for trial in range(EEG['trials']):
                event = {
                    'epoch': trial + 1,  # 1-based for MATLAB compatibility
                    'type': 'TLE',
                    'latency': -EEG['xmin'] * EEG['srate'] + 1 + trial * EEG['pnts']
                }
                EEG['event'].append(event)
        else:
            print('Cannot epoch data with no events')
            return EEG, []

    # Check for latency field
    if not any('latency' in event for event in EEG['event']):
        raise ValueError('Absent latency field in event array/structure: must name one of the fields "latency"')

    # Default parameters
    if types is None:
        types = []
    if lim is None:
        lim = [-1, 2]

    # Process optional arguments
    g = {
        'epochfield': kwargs.get('epochfield', 'type'),  # obsolete
        'timeunit': kwargs.get('timeunit', 'points'),
        'verbose': kwargs.get('verbose', 'on'),  # obsolete
        'newname': kwargs.get('newname', EEG.get('setname', '') + ' epochs' if EEG.get('setname') else ''),
        'eventindices': kwargs.get('eventindices', list(range(len(EEG['event'])))),  # 0-based
        'epochinfo': kwargs.get('epochinfo', 'yes'),
        'valuelim': kwargs.get('valuelim', [-np.inf, np.inf])
    }

    if g['valuelim'] is None:
        g['valuelim'] = [-np.inf, np.inf]

    # Sort events by latency
    tmpevent = copy.deepcopy(EEG['event'])
    tmpeventlatency = [event['latency'] for event in tmpevent]
    sorted_indices = np.argsort(tmpeventlatency)
    EEG['event'] = [EEG['event'][i] for i in sorted_indices]
    
    # Input validation
    if EEG is None:
        raise ValueError('pop_epoch: EEG dataset is required')
    
    # Handle multiple datasets (not implemented)
    if isinstance(EEG, list) and len(EEG) > 1:
        raise NotImplementedError('pop_epoch: multiple datasets not supported')
    
    if isinstance(EEG, list):
        EEG = EEG[0]
    
    # Check for empty event structure
    if 'event' not in EEG or EEG['event'] is None or len(EEG['event']) == 0:
        if EEG.get('trials', 1) > 1 and EEG.get('xmin', 0) <= 0 and EEG.get('xmax', 0) >= 0:
            print("No EEG.event structure found: creating events of type 'TLE' (Time-Locking Event) at time 0")
            # Create TLE events
            EEG['event'] = []
            for trial in range(EEG['trials']):
                event = {
                    'epoch': trial + 1,  # 1-based for MATLAB compatibility
                    'type': 'TLE',
                    'latency': -EEG['xmin'] * EEG['srate'] + 1 + trial * EEG['pnts']
                }
                EEG['event'].append(event)
        else:
            print('Cannot epoch data with no events')
            return EEG, []
    
    # Check for latency field
    if not any('latency' in event for event in EEG['event']):
        raise ValueError('Absent latency field in event array/structure: must name one of the fields "latency"')
    
    # Default parameters
    if types is None:
        types = []
    if lim is None:
        lim = [-1, 2]
    
    # Process optional arguments
    g = {
        'epochfield': kwargs.get('epochfield', 'type'),  # obsolete
        'timeunit': kwargs.get('timeunit', 'points'),
        'verbose': kwargs.get('verbose', 'on'),  # obsolete
        'newname': kwargs.get('newname', EEG.get('setname', '') + ' epochs' if EEG.get('setname') else ''),
        'eventindices': kwargs.get('eventindices', list(range(len(EEG['event'])))),  # 0-based
        'epochinfo': kwargs.get('epochinfo', 'yes'),
        'valuelim': kwargs.get('valuelim', [-np.inf, np.inf])
    }
    
    if g['valuelim'] is None:
        g['valuelim'] = [-np.inf, np.inf]
    
    # Sort events by latency
    tmpevent = copy.deepcopy(EEG['event'])
    tmpeventlatency = [event['latency'] for event in tmpevent]
    sorted_indices = np.argsort(tmpeventlatency)
    EEG['event'] = [EEG['event'][i] for i in sorted_indices]
    
    # Select events for epoching
    Ievent = g['eventindices']
    
    # Load data if needed (assuming data is already loaded)
    if isinstance(EEG.get('data'), str):
        # In MATLAB: EEG = eeg_checkset(EEG, 'loaddata')
        raise NotImplementedError('Data loading from file not implemented')
    
    alllatencies = []
    
    if types:
        # Select events based on types
        if isinstance(types, str):
            # Handle regular expression or simple string
            selected_events = []
            for i, event in enumerate(EEG['event']):
                if i in Ievent:
                    event_type = event.get('type', '')
                    if isinstance(event_type, str):
                        # Use regular expression matching
                        if re.search(types, event_type):
                            selected_events.append(i)
                    elif isinstance(event_type, (int, float)):
                        # Convert to string for comparison
                        if re.search(types, str(event_type)):
                            selected_events.append(i)
        elif isinstance(types, (list, tuple)):
            # Handle list of types
            selected_events = []
            for i, event in enumerate(EEG['event']):
                if i in Ievent:
                    event_type = event.get('type', '')
                    if event_type in types:
                        selected_events.append(i)
        else:
            raise ValueError('pop_epoch: multiple event types must be entered as a list or string')
        
        # Get latencies of selected events
        alllatencies = [EEG['event'][i]['latency'] for i in selected_events]
    else:
        # Use all events
        alllatencies = [EEG['event'][i]['latency'] for i in Ievent]
    
    if not alllatencies:
        raise ValueError('pop_epoch: empty epoch range (no epochs were found)')
    
    print(f'pop_epoch: {len(alllatencies)} epochs selected')
    
    # Convert latencies to appropriate time units
    tmpeventlatency = [event['latency'] for event in EEG['event']]
    
    # Call epoch function
    if g['timeunit'].lower() == 'points':
        # Convert time limits to points
        result = epoch(
            EEG['data'], 
            alllatencies, 
            [lim[0] * EEG['srate'], lim[1] * EEG['srate']], 
            valuelim=g['valuelim'], 
            allevents=tmpeventlatency,
            verbose='off' if g['verbose'] == 'off' else 'on'
        )
        epochdat, tmptime, indices, alleventout, alllatencyout, reallim = result
        tmptime = tmptime / EEG['srate']
    elif g['timeunit'].lower() == 'seconds':
        result = epoch(
            EEG['data'], 
            alllatencies, 
            lim, 
            valuelim=g['valuelim'], 
            srate=EEG['srate'], 
            allevents=tmpeventlatency,
            verbose='off' if g['verbose'] == 'off' else 'on'
        )
        epochdat, tmptime, indices, alleventout, alllatencyout, reallim = result
    else:
        raise ValueError('pop_epoch: invalid event time format')
    
    # Update alllatencies based on accepted epochs
    if types:
        alllatencies = [alllatencies[i] for i in indices]
    else:
        alllatencies = [alllatencies[i] for i in indices]
    
    print(f'pop_epoch: {len(indices)} epochs generated')
    
    # Create output EEG structure
    EEG_out = copy.deepcopy(EEG)
    EEG_out['data'] = epochdat
    
    # Update time fields
    if lim[0] != tmptime[0] or (lim[1] - 1/EEG['srate']) != tmptime[1]:
        print(f'pop_epoch: time limits have been adjusted to [{tmptime[0]:.3f} {tmptime[1] + 1/EEG["srate"]:.3f}] to fit data points limits')
    
    EEG_out['xmin'] = tmptime[0]
    EEG_out['xmax'] = tmptime[1]
    EEG_out['pnts'] = epochdat.shape[1]
    EEG_out['times'] = np.linspace(tmptime[0]*1000, tmptime[1]*1000, epochdat.shape[1])
    EEG_out['trials'] = epochdat.shape[2]
    EEG_out['icaact'] = []  # Clear ICA activations
    
    # Update dataset name and comments
    if EEG.get('setname'):
        if EEG.get('comments'):
            if isinstance(EEG['comments'], str):
                EEG_out['comments'] = f'Parent dataset "{EEG["setname"]}": ----------\n{EEG["comments"]}'
            else:
                EEG_out['comments'] = f'Parent dataset "{EEG["setname"]}": ----------\n' + '\n'.join(EEG['comments'])
        else:
            EEG_out['comments'] = f'Parent dataset: {EEG["setname"]}\n'
    
    EEG_out['setname'] = g['newname']
    
    # Process events for epoched data
    # This is a simplified version - the full MATLAB version is more complex
    if alleventout and len(alleventout) > 0:
        # Count total events to duplicate
        totlen = sum(len(epoch_events) for epoch_events in alleventout)
        
        if totlen > 0:
            newevent = []
            
            # Process events for each epoch
            for epoch_idx in range(len(alleventout)):
                for event_idx in alleventout[epoch_idx]:
                    if event_idx < len(EEG['event']):
                        new_ev = copy.deepcopy(EEG['event'][event_idx])
                        new_ev['epoch'] = epoch_idx + 1  # 1-based epoch numbering
                        
                        # Adjust latency within epoch
                        if types:
                            base_latency = alllatencies[epoch_idx]
                        else:
                            base_latency = alllatencies[epoch_idx]
                        
                        new_ev['latency'] = (new_ev['latency'] - base_latency - 
                                           tmptime[0] * EEG['srate'] + 1 + 
                                           EEG_out['pnts'] * epoch_idx)
                        newevent.append(new_ev)
            
            EEG_out['event'] = newevent
        else:
            EEG_out['event'] = []
    else:
        EEG_out['event'] = []
    
    EEG_out['epoch'] = []  # Clear epoch info
    EEG_out['saved'] = 'no'
    
    # Check for data consistency
    EEG_out = eeg_checkset(EEG_out)
    
    # Check for boundary events
    print('pop_epoch: checking epochs for data discontinuity')
    if EEG_out['event'] is not None and len(EEG_out['event']) > 0:
        if isinstance(EEG_out['event'][0].get('type'), str):
            tmpevent = list(EEG_out['event'])
            boundaryindex = eeg_findboundaries(EEG=tmpevent)
            
            if boundaryindex:
                # Get epochs with boundary events
                indexepoch = []
                for tmpindex in boundaryindex:
                    if 'epoch' in tmpevent[tmpindex]:
                        indexepoch.append(tmpevent[tmpindex]['epoch'])
                    else:
                        indexepoch.append(1)  # only one epoch
                
                # Remove epochs with boundary events
                if indexepoch:
                    EEG_out = pop_select(EEG_out, notrial=indexepoch)
                    if isinstance(EEG_out, tuple) and len(EEG_out) == 2:
                        EEG_out, _ = EEG_out
                    # Update indices of accepted events
                    indices = [idx for idx in indices if (idx + 1) not in indexepoch]  # Convert to 1-based for comparison
    
    return EEG_out, indices
