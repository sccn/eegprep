"""EEG dataset validation and setup utilities."""

import logging
import contextvars
import math
from contextlib import contextmanager

import numpy as np
import os

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.adminfunc.storage import MemmapData, OffloadedData
from eegprep.functions.miscfunc.misc import matlab_mtimes, round_mat

logger = logging.getLogger(__name__)

__all__ = ['eeg_checkset', '_eventconsistency', 'strict_mode', 'option_scaleicarms']

# Global option to control ICA RMS scaling (default True, like MATLAB)
option_scaleicarms = True

# Context variable for strict mode (default True)
_strict_mode_var = contextvars.ContextVar('strict_mode', default=True)


class DummyException(Exception):
    """Exception that should never be raised, used to disable exception handling in strict mode."""

    pass


@contextmanager
def strict_mode(enabled: bool):
    """Context manager to control strict mode for eeg_checkset.

    Args
    ----
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


def _eventconsistency(EEG):
    """Port of MATLAB eeg_checkset 'eventconsistency' event cleanup.

    Performs, in order:
    1. Remove events with NaN latency
    2. Handle first boundary event edge cases (``eeg_isboundary``)
    3. Remove out-of-bounds events (latency < 0.5 or > pnts*trials+1)
    4. Remove events with invalid epoch numbers
    5. Set empty duration fields to 0
    6. Chain durations of consecutive equal-latency boundary-list entries
    7. Sort events by epoch, then latency
    8. Clamp first-event latency to >= 0.5
    """
    from ..popfunc.eeg_findboundaries import eeg_findboundaries

    events = EEG.get('event')
    if events is None:
        return EEG

    if isinstance(events, np.ndarray):
        events = list(events)

    if len(events) == 0:
        return EEG

    events = [event for event in events if not _nan_latency(event)]
    if not events:
        EEG['event'] = np.asarray(events, dtype=object)
        return EEG

    # eeg_checkset.m: a leading boundary with duration < 1 is removed. Otherwise
    # a leading-boundary latency in (0, 1) is clamped to 0.5 before the bounds check.
    if _is_checkset_boundary(events[0]) and 'duration' in events[0]:
        duration = _real_number(events[0].get('duration'))
        if duration is not None and duration < 1:
            events.pop(0)
        else:
            latency = _real_number(events[0].get('latency'))
            if latency is not None and 0 < latency < 1:
                events[0]['latency'] = 0.5

    if not events:
        EEG['event'] = np.asarray(events, dtype=object)
        return EEG

    # 3. Remove out-of-bounds events (MATLAB lines 439-449)
    pnts = int(EEG.get('pnts', 0))
    trials = int(EEG.get('trials', 1))
    upper_bound = pnts * trials + 1
    events = [event for event in events if _latency_in_bounds(event, upper_bound)]
    if not events:
        EEG['event'] = np.asarray(events, dtype=object)
        return EEG

    # 4. Remove events with invalid epoch numbers (MATLAB lines 456-477)
    if trials > 1:
        keep = []
        for event in events:
            epoch = event.get('epoch')
            if epoch is None:
                keep.append(event)
            elif _real_number(epoch) is not None and 1 <= epoch <= trials:
                keep.append(event)
        if len(keep) < len(events):
            logger.info(f"eeg_checkset: {len(events) - len(keep)} events with invalid epoch removed")
        events = keep

    # 5. Set empty/None duration to 0 (MATLAB lines 481-489)
    for event in events:
        if 'duration' in event and (event['duration'] is None or event['duration'] == ''):
            event['duration'] = 0

    # 6. Chain durations of consecutive equal-latency boundary-list entries (MATLAB lines 521-546)
    boundary_idx = eeg_findboundaries(EEG={'event': events, 'setname': '_'})
    if boundary_idx:
        boundary_lats = [events[index].get('latency', float('nan')) for index in boundary_idx]
        # MATLAB round ties away from zero, so round(latency-0.5) > pnts is floor(latency) > pnts.
        if trials == 1 and boundary_lats:
            last_lat = _real_number(boundary_lats[-1])
            if last_lat is not None and math.floor(last_lat) > pnts:
                events.pop(boundary_idx[-1])
                boundary_idx = boundary_idx[:-1]
                boundary_lats = boundary_lats[:-1]

        has_duration = any('duration' in events[index] for index in boundary_idx)
        to_remove = []
        for index in range(1, len(boundary_lats)):
            if boundary_lats[index] != boundary_lats[index - 1]:
                continue
            later = boundary_idx[index]
            earlier = boundary_idx[index - 1]
            if has_duration:
                events[later]['duration'] = _duration_value(events[later]) + _duration_value(events[earlier])
            to_remove.append(earlier)
        if to_remove:
            logger.info("eeg_checkset: duplicate boundary events removed")
            for index in sorted(to_remove, reverse=True):
                events.pop(index)

    # 7. Sort events by epoch, then latency
    events.sort(key=_event_sort_key)

    # 8. Clamp first event latency to >= 0.5  (MATLAB lines 600-604)
    if events and 'latency' in events[0]:
        latency = _real_number(events[0]['latency'])
        if latency is not None and latency < 0.5:
            events[0]['latency'] = 0.5

    EEG['event'] = np.asarray(events, dtype=object)
    return EEG


def _is_checkset_boundary(event):
    """Return whether one event matches ``eeg_isboundary`` (exact, case-insensitive)."""
    event_type = event.get('type')
    if isinstance(event_type, bytes):
        event_type = event_type.decode()
    if isinstance(event_type, str):
        return event_type.lower() == 'boundary'
    event_type = _real_number(event_type)
    return bool(EEG_OPTIONS['option_boundary99']) and event_type == -99


def _real_number(value):
    """Return ``value`` as a finite float, or None for a non-numeric value or NaN.

    Booleans are not numeric. NumPy integers and floats are.
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        return number
    return None


def _nan_latency(event):
    """Return whether the event has a floating-point NaN latency."""
    latency = event.get('latency')
    return 'latency' in event and isinstance(latency, (float, np.floating)) and np.isnan(latency)


def _latency_in_bounds(event, upper_bound):
    """Return whether latency is absent or inside ``[0.5, upper_bound]``.

    A NaN latency is out of bounds. An event with no latency field is kept.
    """
    if 'latency' not in event:
        return True
    if _nan_latency(event):
        return False
    latency = event['latency']
    return latency >= 0.5 and latency <= upper_bound


def _duration_value(event):
    """Return the event duration as a float, or 0 when it is missing or not numeric."""
    duration = _real_number(event.get('duration'))
    if duration is None:
        return 0.0
    return duration


def _event_sort_key(event):
    """Return ``(epoch, latency)`` so events sort by epoch, then latency.

    A missing or non-numeric field sorts as 0.
    """
    epoch = _real_number(event.get('epoch', 0))
    latency = _real_number(event.get('latency', 0))
    return (0 if epoch is None else epoch, 0 if latency is None else latency)


def _refresh_icaact(EEG) -> None:
    """Recompute stored activations from the current data array."""
    if (
        'icaweights' not in EEG
        or 'icasphere' not in EEG
        or not hasattr(EEG['icaweights'], 'size')
        or not hasattr(EEG['icasphere'], 'size')
        or EEG['icaweights'].size == 0
        or EEG['icasphere'].size == 0
    ):
        return
    if (
        'icachansind' in EEG
        and EEG['icachansind'] is not None
        and hasattr(EEG['icachansind'], '__len__')
        and len(EEG['icachansind']) > 0
    ):
        icachansind = np.array(EEG['icachansind'], dtype=int)
        n_channels = EEG['data'].shape[0]
        if np.any(icachansind < 0) or np.any(icachansind >= n_channels):
            logger.warning(
                f"icachansind contains out-of-bounds indices (valid range: 0-{n_channels - 1}). "
                "Using all channels instead."
            )
            ica_data = EEG['data']
        else:
            ica_data = EEG['data'][icachansind, :]
    else:
        ica_data = EEG['data']

    expected_channels = EEG['icaweights'].shape[1]
    actual_channels = ica_data.shape[0]
    if expected_channels != actual_channels:
        logger.warning(
            f"ICA dimension mismatch: icaweights expects {expected_channels} channels "
            f"but ica_data has {actual_channels} channels. Skipping ICA activation computation."
        )
        EEG['icaact'] = np.array([])
        return
    EEG['icaact'] = _ica_activation_matrix(EEG, ica_data)


def _ica_activation_matrix(EEG, ica_data):
    """Return ``(icaweights*icasphere)*data`` in the data class.

    ``eeg_checkset.m:1030`` multiplies column-major, so a single ``data`` array
    uses the single-precision GEMM. ``reshape`` at line 1032 drops a trailing
    singleton when ``trials == 1``.
    """
    flat = np.asarray(ica_data).reshape(ica_data.shape[0], -1, order="F")
    unmixing = matlab_mtimes(np.asarray(EEG["icaweights"]), np.asarray(EEG["icasphere"]))
    product = matlab_mtimes(unmixing, flat)
    components = EEG["icaweights"].shape[0]
    pnts = int(EEG["pnts"])
    trials = int(EEG["trials"])
    if trials == 1:
        return np.reshape(product, (components, pnts), order="F")
    return np.reshape(product, (components, pnts, trials), order="F")


def eeg_checkset(EEG, *checks, load_data=True):
    """Validate and set up EEG dataset structure.

    Ensures EEG dict has required fields with correct types, computes ICA activations if
    possible, and loads data from file if specified.
    """
    # In strict mode (True), we catch DummyException (never raised) so exceptions propagate
    # In non-strict mode (False), we catch Exception and handle gracefully
    exception_type = DummyException if _strict_mode_var.get() else Exception

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
        file_name = EEG['filepath'] + os.sep + EEG['data']
        if not os.path.exists(file_name):
            # try to use the same name as the filename but with .fdt extension
            file_name = EEG['filepath'] + os.sep + EEG['filename'].replace('.set', '.fdt')
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"Data file {file_name} not found")
        EEG['data'] = np.fromfile(file_name, dtype='float32').reshape(EEG['pnts'] * EEG['trials'], EEG['nbchan'])
        EEG['data'] = EEG['data'].T.reshape(EEG['nbchan'], EEG['trials'], EEG['pnts']).transpose(0, 2, 1)

    # Scale ICA components to RMS microvolt (matches MATLAB's option_scaleicarms)
    if (
        option_scaleicarms
        and 'icaweights' in EEG
        and 'icasphere' in EEG
        and 'icawinv' in EEG
        and hasattr(EEG['icaweights'], 'size')
        and hasattr(EEG['icasphere'], 'size')
        and hasattr(EEG['icawinv'], 'size')
        and EEG['icaweights'].size > 0
        and EEG['icasphere'].size > 0
        and EEG['icawinv'].size > 0
    ):
        try:
            # Check if pinv(icaweights @ icasphere) ≈ icawinv
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                computed_icawinv = np.linalg.pinv(EEG['icaweights'] @ EEG['icasphere'])
            mean_diff = np.mean(np.abs(computed_icawinv - EEG['icawinv']))

            if mean_diff < 0.0001:
                logger.info('Scaling components to RMS microvolt')
                # Compute RMS of each column of icawinv (each component's scalp projection)
                # scaling shape: (n_components,)
                scaling = np.sqrt(np.mean(EEG['icawinv'] ** 2, axis=0))

                if not isinstance(EEG.get('etc'), dict):
                    logger.debug("Replacing non-dict EEG['etc'] during ICA RMS scaling backup")
                    EEG['etc'] = {}
                EEG['etc']['icaweights_beforerms'] = EEG['icaweights'].copy()
                EEG['etc']['icasphere_beforerms'] = EEG['icasphere'].copy()

                # Apply scaling to icaweights (each row scaled by corresponding component's RMS)
                # icaweights shape: (n_components, n_channels)
                EEG['icaweights'] = EEG['icaweights'] * scaling[:, np.newaxis]

                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    EEG['icawinv'] = np.linalg.pinv(EEG['icaweights'] @ EEG['icasphere'])
        except exception_type as e:
            logger.error("Error scaling ICA components: " + str(e))

    try:
        _refresh_icaact(EEG)
    except exception_type as e:
        logger.error("Error computing ICA activations: " + str(e))
        EEG['icaact'] = np.array([])

    # Handle optional check modes (matches MATLAB's eeg_checkset(EEG, 'mode') API)
    for check in checks:
        if isinstance(check, str) and check.lower() == 'eventconsistency':
            EEG = _eventconsistency(EEG)

    # Build epoch structure from events (for epoched data)
    # This matches MATLAB's eeg_checkset behavior (lines 611-670 in eeg_checkset.m)
    try:
        if EEG.get('trials', 1) > 1 and len(EEG.get('event', [])) > 0:
            if 'epoch' not in EEG:
                EEG['epoch'] = []

            if len(EEG['epoch']) != EEG['trials']:
                # Need to rebuild epoch structure
                EEG['epoch'] = []
            elif len(EEG['epoch']) > 0:
                # (they will be regenerated from current events)
                epoch_list = list(EEG['epoch'])
                for i in range(len(epoch_list)):
                    if isinstance(epoch_list[i], dict):
                        keys_to_remove = [k for k in epoch_list[i].keys() if k.startswith('event')]
                        for k in keys_to_remove:
                            del epoch_list[i][k]
                EEG['epoch'] = np.array(epoch_list, dtype=object)

            tmpevent = EEG['event']
            eventepoch = np.array([e.get('epoch', 1) for e in tmpevent])

            epochevent = []
            for k in range(EEG['trials']):
                epoch_num = k + 1  # 1-based epoch numbering
                event_indices = np.where(eventepoch == epoch_num)[0]
                epochevent.append(list(event_indices))

            if len(EEG['epoch']) == 0:
                EEG['epoch'] = np.array([{} for _ in range(EEG['trials'])], dtype=object)

            # Set event field in each epoch
            for k in range(EEG['trials']):
                EEG['epoch'][k]['event'] = epochevent[k]

            # Copy event information into the epoch array
            # Skip 'epoch' field itself to avoid duplication
            event_fields = []
            if len(tmpevent) > 0:
                event_fields = [f for f in tmpevent[0].keys() if f != 'epoch']

            for fname in event_fields:
                for k in range(EEG['trials']):
                    event_idx_list = epochevent[k]
                    if len(event_idx_list) == 0:
                        # No events in this epoch
                        field_values = []
                    else:
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

                    EEG['epoch'][k][f'event{fname}'] = field_values

    except exception_type as e:
        logger.warning(f"Could not build epoch structure: {e}")

    # Single-trial data carries no epoch structure and no event.epoch field
    # (eeg_checkset.m, "check if only one epoch").
    if int(EEG.get('trials', 1) or 1) == 1:
        for ev in EEG['event']:
            ev.pop('epoch', None)
        EEG['epoch'] = np.array([], dtype=object)

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

    if 'xmin' in EEG and int(EEG.get('trials', 1) or 1) == 1 and EEG['xmin'] != 0:
        EEG['xmin'] = 0.0
        logger.info('eeg_checkset: xmin set to 0 for continuous data')

    if all(key in EEG for key in ('srate', 'xmin', 'xmax', 'pnts')):
        if EEG['srate'] == 0:
            EEG['srate'] = 1.0
        # eeg_checkset.m:849. Half away from zero, so an even n+0.5 rounds up.
        expected_pnts = int(round_mat(EEG['srate'] * (EEG['xmax'] - EEG['xmin']) + 1))
        if expected_pnts != EEG['pnts']:
            EEG['xmax'] = (EEG['pnts'] - 1) / EEG['srate'] + EEG['xmin']
            logger.info('eeg_checkset: xmax adjusted to match pnts, xmin, and srate')
        if EEG['pnts'] > 0:
            EEG['times'] = np.linspace(EEG['xmin'] * 1000, EEG['xmax'] * 1000, EEG['pnts'])

    expected_types = {
        'setname': str,
        'filename': str,
        'filepath': str,
        'subject': str,
        'group': str,
        'condition': str,
        'session': (str, int),
        'comments': (str, np.ndarray),
        'nbchan': int,
        'trials': int,
        'pnts': int,
        'srate': (float, int),
        'xmin': float,
        'xmax': float,
        'times': np.ndarray,  # Expecting a float numpy array
        'data': np.ndarray,  # Expecting a float numpy array
        'icaact': np.ndarray,  # Expecting a float numpy array
        'icawinv': np.ndarray,  # Expecting a float numpy array
        'icasphere': np.ndarray,  # Expecting a float numpy array
        'icaweights': np.ndarray,  # Expecting a float numpy array
        'icachansind': np.ndarray,  # Expecting an integer numpy array
        'chanlocs': np.ndarray,  # Expecting numpy array of dictionaries
        'urchanlocs': np.ndarray,  # Expecting numpy array of dictionaries
        'chaninfo': dict,
        'ref': str,
        'event': np.ndarray,  # Expecting numpy array of dictionaries
        'urevent': np.ndarray,  # Expecting numpy array of dictionaries
        'eventdescription': np.ndarray,  # Expecting numpy array of strings
        'epoch': np.ndarray,  # Expecting numpy array of dictionaries
        'epochdescription': np.ndarray,  # Expecting numpy array of strings
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

    for field, expected_type in expected_types.items():
        if field not in EEG:
            logger.debug(f"Field '{field}' is missing from the EEG dictionary, adding it.")

            # add default values
            if expected_type is str:
                EEG[field] = ''
            elif expected_type is int:
                EEG[field] = np.array([], dtype=int)
            elif expected_type is float:
                EEG[field] = np.array([], dtype=float)
            elif expected_type is dict:
                EEG[field] = {}
            elif expected_type == np.ndarray:
                EEG[field] = np.array([])
            else:
                EEG[field] = np.array([])
            continue

        value = EEG[field]

        # Special cases for numpy arrays with specific content types
        if isinstance(expected_type, type) and expected_type == np.ndarray:
            if field == 'data' and isinstance(value, (MemmapData, OffloadedData)):
                continue
            if not isinstance(value, np.ndarray):
                logger.warning(
                    f"Field '{field}' is expected to be a numpy array but is of type {type(value).__name__}."
                )
                continue
            # Further checks for numpy array content types
            if field in ['times', 'data', 'icaact', 'icawinv', 'icasphere', 'icaweights']:
                if not np.issubdtype(value.dtype, np.floating):
                    logger.warning(
                        f"Field '{field}' is expected to be a numpy array of floats but has dtype {value.dtype}."
                    )
            elif field in ['icachansind']:
                if not np.issubdtype(value.dtype, np.integer):
                    logger.warning(
                        f"Field '{field}' is expected to be a numpy array of integers but has dtype {value.dtype}."
                    )
            elif field in ['chanlocs', 'urchanlocs', 'event', 'urevent', 'epoch']:
                if not all(isinstance(item, dict) for item in value):
                    logger.warning(
                        f"Field '{field}' is expected to be a numpy array of dictionaries but contains other types."
                    )
        else:
            # General type check
            if not isinstance(value, expected_type):
                if isinstance(value, np.ndarray) and value.size == 0:
                    continue
                logger.warning(
                    f"Field '{field}' is expected to be of type {expected_type} but is of type {type(value).__name__}."
                )

    return EEG
