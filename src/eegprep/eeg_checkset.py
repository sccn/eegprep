
import logging
import contextvars
from contextlib import contextmanager

# test_eeg_checkset()
import numpy as np
import os

logger = logging.getLogger(__name__)

__all__ = ['eeg_checkset', 'strict_mode', 'option_scaleicarms']

# Global option to control ICA RMS scaling (default True, like MATLAB)
option_scaleicarms = True

# Context variable for strict mode (default True)
_strict_mode_var = contextvars.ContextVar('strict_mode', default=True)

class DummyException(Exception):
    """Exception that should never be raised, used to disable exception handling in strict mode"""
    pass

@contextmanager
def strict_mode(enabled: bool):
    """
    Context manager to control strict mode for eeg_checkset.
    
    Args:
        enabled (bool): If True, exceptions will propagate (strict mode).
                       If False, exceptions will be caught and handled gracefully.
    
    Usage:
        with strict_mode(False):
            EEG = eeg_checkset(EEG)  # Will catch and handle exceptions
    """
    token = _strict_mode_var.set(enabled)
    try:
        yield
    finally:
        _strict_mode_var.reset(token)


def eeg_checkset(EEG, load_data=True):
    # Get the exception type based on strict mode
    # In strict mode (True), we catch DummyException (never raised) so exceptions propagate
    # In non-strict mode (False), we catch Exception and handle gracefully
    exception_type = DummyException if _strict_mode_var.get() else Exception
        

    # convert EEG['nbchan] to integer
    if 'nbchan' in EEG:
        EEG['nbchan'] = int(EEG['nbchan'])
    else:
        EEG['nbchan'] = EEG['data'].shape[0]
    if 'pnts' in EEG:
        EEG['pnts'] = int(EEG['pnts'])
    else:
        EEG['pnts'] = EEG['data'].shape[1]
    if 'trials' in EEG:
        EEG['trials'] = int(EEG['trials'])
    else:
        if EEG['data'].ndim == 3:
            EEG['trials'] = EEG['data'].shape[2]
        else:
            EEG['trials'] = 1
            
    if 'event' in EEG:
        if isinstance(EEG['event'], dict):
            EEG['event'] = [EEG['event']]
    else:
        EEG['event'] = []
    if isinstance(EEG['event'], list):
        EEG['event'] = np.asarray(EEG['event'], dtype=object)
            
    if 'chanlocs' in EEG:
        if isinstance(EEG['chanlocs'], dict):
            EEG['chanlocs'] = [EEG['chanlocs']]
    else:
        EEG['chanlocs'] = []
    if isinstance(EEG['chanlocs'], list):
        EEG['chanlocs'] = np.asarray(EEG['chanlocs'], dtype=object)

    if 'chaninfo' not in EEG:
        EEG['chaninfo'] = {}
        
    if 'reject' not in EEG:
        EEG['reject'] = {}
        
    if 'data' in EEG and isinstance(EEG['data'], str) and load_data:
        # get path from file_path
        file_name = EEG['filepath'] + os.sep + EEG['data']
        if not os.path.exists(file_name):
            # try to use the sane name as the filename but with .fdt extension
            file_name = EEG['filepath'] + os.sep + EEG['filename'].replace('.set', '.fdt')
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"Data file {file_name} not found")
        EEG['data'] = np.fromfile(file_name, dtype='float32').reshape( EEG['pnts']*EEG['trials'], EEG['nbchan'])
        EEG['data'] = EEG['data'].T.reshape(EEG['nbchan'], EEG['trials'], EEG['pnts']).transpose(0, 2, 1)

    # Scale ICA components to RMS microvolt (matches MATLAB's option_scaleicarms)
    if (option_scaleicarms and
        'icaweights' in EEG and 'icasphere' in EEG and 'icawinv' in EEG and
        hasattr(EEG['icaweights'], 'size') and hasattr(EEG['icasphere'], 'size') and
        hasattr(EEG['icawinv'], 'size') and
        EEG['icaweights'].size > 0 and EEG['icasphere'].size > 0 and EEG['icawinv'].size > 0):
        
        try:
            # Check if pinv(icaweights @ icasphere) ≈ icawinv
            computed_icawinv = np.linalg.pinv(EEG['icaweights'] @ EEG['icasphere'])
            mean_diff = np.mean(np.abs(computed_icawinv - EEG['icawinv']))
            
            if mean_diff < 0.0001:
                logger.info('Scaling components to RMS microvolt')
                # Compute RMS of each column of icawinv (each component's scalp projection)
                # scaling shape: (n_components,)
                scaling = np.sqrt(np.mean(EEG['icawinv'] ** 2, axis=0))
                
                # Store original weights before scaling
                if 'etc' not in EEG:
                    EEG['etc'] = {}
                EEG['etc']['icaweights_beforerms'] = EEG['icaweights'].copy()
                EEG['etc']['icasphere_beforerms'] = EEG['icasphere'].copy()
                
                # Apply scaling to icaweights (each row scaled by corresponding component's RMS)
                # icaweights shape: (n_components, n_channels)
                EEG['icaweights'] = EEG['icaweights'] * scaling[:, np.newaxis]
                
                # Recompute icawinv
                EEG['icawinv'] = np.linalg.pinv(EEG['icaweights'] @ EEG['icasphere'])
        except exception_type as e:
            logger.error("Error scaling ICA components: " + str(e))

    # compute ICA activations
    if ('icaweights' in EEG and 'icasphere' in EEG and 
        hasattr(EEG['icaweights'], 'size') and hasattr(EEG['icasphere'], 'size') and
        EEG['icaweights'].size > 0 and EEG['icasphere'].size > 0):
        
        try:
            EEG['icaact'] = np.dot(np.dot(EEG['icaweights'], EEG['icasphere']), EEG['data'].reshape(int(EEG['nbchan']), -1))
            EEG['icaact'] = EEG['icaact'].astype(np.float32)
            EEG['icaact'] = EEG['icaact'].reshape(EEG['icaweights'].shape[0], -1, int(EEG['trials']))
        except exception_type as e:
            logger.error("Error computing ICA activations: " + str(e))
            EEG['icaact'] = np.array([])
    
    # Build epoch structure from events (for epoched data)
    # This matches MATLAB's eeg_checkset behavior (lines 611-670 in eeg_checkset.m)
    try:
        if EEG.get('trials', 1) > 1 and len(EEG.get('event', [])) > 0:
            # Initialize epoch field if missing or wrong size
            if 'epoch' not in EEG:
                EEG['epoch'] = []

            if len(EEG['epoch']) != EEG['trials']:
                # Need to rebuild epoch structure
                EEG['epoch'] = []
            elif len(EEG['epoch']) > 0:
                # Remove existing event-related fields from epoch structure
                # (they will be regenerated from current events)
                epoch_list = list(EEG['epoch'])
                for i in range(len(epoch_list)):
                    if isinstance(epoch_list[i], dict):
                        keys_to_remove = [k for k in epoch_list[i].keys() if k.startswith('event')]
                        for k in keys_to_remove:
                            del epoch_list[i][k]
                EEG['epoch'] = np.array(epoch_list, dtype=object)

            # Build epoch structure from events
            tmpevent = EEG['event']
            eventepoch = np.array([e.get('epoch', 1) for e in tmpevent])

            # Create list of event indices for each epoch
            epochevent = []
            for k in range(EEG['trials']):
                epoch_num = k + 1  # 1-based epoch numbering
                event_indices = np.where(eventepoch == epoch_num)[0]
                epochevent.append(list(event_indices))

            # Initialize epoch structure if empty
            if len(EEG['epoch']) == 0:
                EEG['epoch'] = np.array([{} for _ in range(EEG['trials'])], dtype=object)

            # Set event field in each epoch
            for k in range(EEG['trials']):
                EEG['epoch'][k]['event'] = epochevent[k]

            # Copy event information into the epoch array
            # Skip 'epoch' field itself to avoid duplication
            event_fields = []
            if len(tmpevent) > 0:
                # Get field names from first event
                event_fields = [f for f in tmpevent[0].keys() if f != 'epoch']

            for fname in event_fields:
                # Build list of field values for each epoch
                for k in range(EEG['trials']):
                    event_idx_list = epochevent[k]
                    if len(event_idx_list) == 0:
                        # No events in this epoch
                        field_values = []
                    else:
                        # Extract values from events for this epoch
                        field_values = []
                        for idx in event_idx_list:
                            val = tmpevent[idx].get(fname, None)

                            # Special handling for latency: convert to epoch-relative time in ms
                            if fname == 'latency' and val is not None:
                                # Convert from absolute sample to epoch-relative time in ms
                                # Formula from MATLAB: eeg_point2lat(latency, epoch, srate, [xmin xmax]*1000, 1e-3)
                                epoch_start_sample = k * EEG['pnts']
                                rel_sample = val - epoch_start_sample - 1  # -1 because latencies are 1-based
                                rel_time_ms = (rel_sample / EEG['srate']) * 1000 + EEG['xmin'] * 1000
                                val = round(rel_time_ms * 1e8) / 1e8  # Round to match MATLAB precision

                            # Special handling for duration: convert to ms
                            elif fname == 'duration' and val is not None:
                                val = val / EEG['srate'] * 1000

                            field_values.append(val)

                    # Store in epoch structure
                    EEG['epoch'][k][f'event{fname}'] = field_values

    except exception_type as e:
        logger.warning(f"Could not build epoch structure: {e}")

    # check if EEG['data'] is 3D
    if 'data' in EEG and EEG['data'].ndim == 3:
        if EEG['data'].shape[2] == 1:
            EEG['data'] = np.squeeze(EEG['data'], axis=2)
     
    # type conversion (only if fields exist)
    if 'xmin' in EEG:
        EEG['xmin'] = float(EEG['xmin'])
    if 'xmax' in EEG:
        EEG['xmax'] = float(EEG['xmax'])
    if 'srate' in EEG:
        EEG['srate'] = float(EEG['srate'])
         
    # Define the expected types
    expected_types = {
        'setname': str,
        'filename': str,
        'filepath': str,
        'subject': str,
        'group': str,
        'condition': str,
        'session': (str, int),
        'comments': np.ndarray,
        'nbchan': int,
        'trials': int,
        'pnts': int,
        'srate': (float,int),
        'xmin': float,
        'xmax': float,
        'times': np.ndarray,  # Expecting a float numpy array
        'data': np.ndarray,   # Expecting a float numpy array
        'icaact': np.ndarray, # Expecting a float numpy array
        'icawinv': np.ndarray,# Expecting a float numpy array
        'icasphere': np.ndarray, # Expecting a float numpy array
        'icaweights': np.ndarray, # Expecting a float numpy array
        'icachansind': np.ndarray, # Expecting an integer numpy array
        'chanlocs': np.ndarray,    # Expecting numpy array of dictionaries
        'urchanlocs': np.ndarray,  # Expecting numpy array of dictionaries
        'chaninfo': dict,
        'ref': str,
        'event': np.ndarray,       # Expecting numpy array of dictionaries
        'urevent': np.ndarray,     # Expecting numpy array of dictionaries
        'eventdescription': np.ndarray, # Expecting numpy array of strings
        'epoch': np.ndarray,       # Expecting numpy array of dictionaries
        'epochdescription': np.ndarray, # Expecting numpy array of strings
        'reject': dict,
        'stats': dict,
        'specdata': dict,
        'specicaact': dict,
        'splinefile': str,
        'icasplinefile': str,
        'dipfit': dict,
        'history': str,
        'saved': str,
        'etc': dict,
        'datfile': str,
        'run': (str, int),
        'roi': dict,
    }
    
    # Iterate through expected types and check input dictionary
    for field, expected_type in expected_types.items():
        if field not in EEG:
            print(f"Field '{field}' is missing from the EEG dictionnary, adding it.")
            
            # add default values
            if expected_type == str:
                EEG[field] = ''
            elif expected_type == int:
                EEG[field] = np.array([], dtype=int)
            elif expected_type == float:
                EEG[field] = np.array([], dtype=float)
            elif expected_type == dict:
                EEG[field] = {}
            elif expected_type == np.ndarray:
                EEG[field] = np.array([])
            else:
                EEG[field] = np.array([])
            continue
        
        value = EEG[field]
        
        # Special cases for numpy arrays with specific content types
        if isinstance(expected_type, type) and expected_type == np.ndarray:
            if not isinstance(value, np.ndarray):
                print(f"Field '{field}' is expected to be a numpy array but is of type {type(value).__name__}.")
                continue
            # Further checks for numpy array content types
            if field in ['times', 'data', 'icaact', 'icawinv', 'icasphere', 'icaweights']:
                if not np.issubdtype(value.dtype, np.floating):
                    print(f"Field '{field}' is expected to be a numpy array of floats but has dtype {value.dtype}.")
            elif field in ['icachansind']:
                if not np.issubdtype(value.dtype, np.integer):
                    print(f"Field '{field}' is expected to be a numpy array of integers but has dtype {value.dtype}.")
            elif field in ['chanlocs', 'urchanlocs', 'event', 'urevent', 'epoch']:
                if not all(isinstance(item, dict) for item in value):
                    print(f"Field '{field}' is expected to be a numpy array of dictionaries but contains other types.")
            # elif field in ['eventdescription', 'epochdescription']:
            #     if not all(isinstance(item, str) for item in value):
            #         print(f"Field '{field}' is expected to be a numpy array of strings but contains other types.")
        else:
            # General type check
            if not isinstance(value, expected_type):
                # check for empty Ndarray
                if isinstance(value, np.ndarray) and value.size == 0:
                    continue
                print(f"Field '{field}' is expected to be of type {expected_type} but is of type {type(value).__name__}.")  
    
    return EEG

def test_eeg_checkset():
    from eegprep.pop_loadset import pop_loadset

    eeglab_file_path = './data/eeglab_data_with_ica_tmp_out2.set'
    EEG = pop_loadset(eeglab_file_path)
    EEG = eeg_checkset(EEG)
    print('Checkset done')

if __name__ == '__main__':
    test_eeg_checkset()