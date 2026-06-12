"""Module for loading EEG data from BIDS datasets."""

import os
import copy
from typing import Dict, Any, Tuple, Union, Optional
import logging
import warnings
from eegprep.plugins.EEG_BIDS.bids import layout_for_fpath, layout_get_lenient, query_for_adjacent_fpath, root_for_fpath
from eegprep.plugins.EEG_BIDS.coords import (
    chanloc_has_coords,
    chanlocs_to_coords,
    clear_chanloc,
    coords_ALS_to_angular,
    coords_RAS_to_ALS,
    coords_to_mm,
)
from eegprep.plugins.EEG_BIDS.montage import apply_montage_inference
from eegprep.plugins.EEG_BIDS.raw import load_raw_eeg_file
from eegprep.functions.miscfunc.misc import ExceptionUnlessDebug, round_mat

import numpy as np

logger = logging.getLogger(__name__)


# list of candidate column names for event types in BIDS events files, in order of preference.
event_type_columns = ['trial_type', 'type', 'event_type', 'HED', 'value', 'code']

# a list of column names that we interpret to contain event timing information
event_timing_columns = ['onset', 'duration', 'sample']


def pop_load_frombids(
    filename: str,
    *,
    bidsmetadata: bool = True,
    bidschanloc: bool = True,
    bidsevent: Union[bool, str] = 'replace',
    eventtype: Optional[str] = None,
    infer_locations: Union[bool, str, None] = None,
    dtype: np.dtype = np.float32,
    numeric_null: Any = np.array([]),
    return_report: bool = False,
    verbose: bool = True,
) -> Dict[str, Any] | Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load an EEG data file of a supported format from a BIDS dataset.

    Supported formats are EDF, BrainVision, EEGLAB SET, BDF.

    Parameters
    ----------
    filename : str
        Path to the EEG data file in a BIDS dataset.
    bidsmetadata : bool
        Whether to override any metadata in the EEG file with
        metadata from BIDS.
    bidschanloc : bool
        Whether to override any channel information (incl. locations)
        in the EEG file with channel information from BIDS.
    bidsevent : bool or str
        Whether to load in and override any event data in the EEG file with
        event data from BIDS. Can be one of the following:

        * ``"replace"``/``True``: replace events from EEG file with those from
          the BIDS event file.
        * ``"merge"``: selectively override events from EEG file with those
          from the BIDS event file.
        * ``"append"``: append events from the BIDS event file to those from
          the EEG file. This mode can result in duplicate events; use with
          caution.
        * ``False``/``None``: do not load events from BIDS; keep those from the
          EEG file.
    eventtype : str or None
        Optionally the column name in the BIDS events file to use for event
        types; if not set, will be inferred heuristically.
    infer_locations : bool or str or None
        Whether to infer channel locations if necessary from the
        channel labels (if 10-20 labeling system).

        * ``True``: infer locations from channel labels and override existing
          locations if any.
        * ``False``: leave locations as-is, even if missing.
        * ``None``: infer only if no channels have locations.
        * ``str``: filename of a locations file to infer locations from. See
          files in ``resources/montages``; this can disambiguate alternative
          montages that use the same naming system.
    dtype : np.dtype
        The data type to use for the EEG data.
    numeric_null : Any
        The value to use for empty numeric fields in the EEG data.

        The default is ``np.array([])`` for MATLAB/pop_loadset compatibility.
    return_report : bool
        whether to return an import report dictionary as a second output
    verbose : bool
        whether to log verbose output

    Returns
    -------
    EEG : dict
        A dictionary containing the EEG data and metadata.
    Report : dict, optional
        optionally the import report to return, if desired.
    """
    from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset

    report = {
        'Warnings': [],
        'Errors': [],
    }

    def warning(msg: str):
        logger.warning(msg)
        report['Warnings'].append(msg)

    def error(msg: str):
        logger.error(msg)
        report['Errors'].append(msg)

    _path, ext = os.path.splitext(filename)
    ext = ext.lower()

    root = root_for_fpath(filename)

    if verbose:
        logger.info(f"Loading EEG data from {filename}...")
    basename = os.path.basename(filename)
    EEG, Fs, times_sec, raw_report = load_raw_eeg_file(
        filename,
        dtype=dtype,
        numeric_null=numeric_null,
        warning=warning,
        verbose=verbose,
    )
    report.update(raw_report)

    report['EEGFileHadLocations'] = sum(chanloc_has_coords(ch) for ch in EEG['chanlocs'])
    report['ChanlocsFrom'] = os.path.relpath(filename, root)
    report['EEGFileHadEvents'] = len(EEG['event'])
    report['EventsFrom'] = os.path.relpath(filename, root)

    if bidsmetadata:
        if verbose:
            logger.info("  applying BIDS metadata...")
        import bids

        layout: bids.BIDSLayout = layout_for_fpath(filename)
        # get the applicable metadata for this file
        metadata: Dict[str, Any] = layout.get_metadata(filename, include_entities=True)

        # apply overrides
        EEG['subject'] = metadata.get('subject', '')
        if EEG['ref'] == 'unknown':
            EEG['ref'] = metadata.get('EEGReference', 'unknown')
        EEG['etc']['BIDS'] = metadata

    if bidschanloc:
        import bids

        layout: bids.BIDSLayout = layout_for_fpath(filename)

        # check for presence of a _channels.tsv file
        query_entities = {**layout.parse_file_entities(filename), 'suffix': 'channels', 'extension': '.tsv'}
        # retrieve the list of all such files
        channel_file_list = layout_get_lenient(
            layout,
            **query_entities,
            return_type='object',
            tolerate_missing=('task', 'run'),
            expect_one=True,
        )
        if len(channel_file_list) > 1:
            warning(
                f"Found multiple BIDS channel files for {filename}: "
                f"{', '.join([fo.filename for fo in channel_file_list])}. "
                f"Using the first one only."
            )
        for fo in channel_file_list:
            import pandas as pd

            if verbose:
                logger.info(f"  applying BIDS channel locations from {fo.filename}...")
            report['ChanlocsFrom'] = os.path.relpath(fo.path, root)
            # read in the file contents
            chans: pd.DataFrame = fo.get_df()

            # this is used to override the type (e.g. 'EEG', 'EOG', 'ECG', etc.) and the ref (if present)
            notfound = []
            notype = False
            has_ref = 'reference' in chans.columns
            orig_labels = [cl['labels'] for cl in EEG['chanlocs']]
            for ch in chans.iloc:
                lab = ch['name']
                if lab not in orig_labels:
                    notfound.append(lab)
                    continue
                idx = orig_labels.index(lab)

                # update the channel type
                try:
                    typ = ch['type']
                except KeyError:
                    notype = True
                else:
                    EEG['chanlocs'][idx]['type'] = typ

                # update the reference, if present
                if has_ref:
                    try:
                        ref_idx = orig_labels.index(ch['reference'])
                    except ValueError:
                        # perhaps best to just leave it as is since the EEG file
                        # might have it set already
                        # EEG['chanlocs'][idx]['ref'] = numeric_null
                        pass
                    else:
                        EEG['chanlocs'][idx]['ref'] = ref_idx
            if notfound:
                nf = [str(n) for n in notfound]
                warning(
                    f"Channels {','.join(nf)} from BIDS file {fo.filename} not found in EEG data structure; skipping."
                )
            if notype:
                warning(f"Channels in BIDS file {fo.filename} do not have a 'type' column; not overriding.")

            break

        # check for presence of an _electrodes.tsv file
        query_entities = query_for_adjacent_fpath(filename, suffix='electrodes', extension='.tsv')
        # retrieve the list of all such files
        elec_file_list = layout_get_lenient(
            layout,
            **query_entities,
            return_type='object',
            tolerate_missing=('task', 'run'),
            expect_one=True,
        )
        if len(elec_file_list) > 1:
            warning(
                f"Found multiple BIDS electrode files for {filename}: "
                f"{', '.join([fo.filename for fo in elec_file_list])}. "
                f"Using the first one only."
            )
            elec_file_list = elec_file_list[:1]
        for elec_fo in elec_file_list:
            import pandas as pd

            if verbose:
                logger.info(f"  applying BIDS electrode locations from {fo.filename}...")
            report['ElectrodesFrom'] = os.path.relpath(elec_fo.path, root)
            # read in the file contents
            elecs: pd.DataFrame = elec_fo.get_df()

            # check for the presence of a coordsystem file
            query_entities = query_for_adjacent_fpath(elec_fo.path, suffix='coordsystem', extension='.json')
            coordsystem_file_list = layout_get_lenient(
                layout,
                **query_entities,
                return_type='object',
                tolerate_missing=('task', 'run', 'space'),
                expect_one=True,
            )
            if len(coordsystem_file_list) > 1:
                warning(
                    f"Found multiple BIDS coordsystem files for {elec_fo.filename}: "
                    f"{', '.join([fo.filename for fo in coordsystem_file_list])}. "
                    f"Using the first one only."
                )
                coordsystem_file_list = coordsystem_file_list[:1]
            if not coordsystem_file_list:
                # if it's a .set study, then we assume ALS for the chanlocs, otherwise RAS
                coord_system = 'ALS' if ext == '.set' else 'RAS'
                coord_units = 'guess'
                warning(
                    f"Found no BIDS coordsystem files for {fo.filename}; your "
                    f"dataset is not fully BIDS-compliant. Assuming coordinate "
                    f"system {coord_system!r} and guessing units from the data."
                )
            else:
                for coordsystem_fo in coordsystem_file_list:
                    if verbose:
                        logger.info(f"  applying BIDS coordsystem from {coordsystem_fo.filename}...")
                    report['CoordsystemFrom'] = os.path.relpath(coordsystem_fo.path, root)
                    # read in the file contents
                    content: Dict[str, Any] = coordsystem_fo.get_dict()
                    EEG['etc']['BIDSCoordsystem'] = content
                    coord_system = content.get('EEGCoordinateSystem', 'RAS')  # default to RAS if not specified
                    if 'EEGLAB' in coord_system.upper():
                        # as per BIDS docs, EEGLAB is the only one that's expressly not RAS
                        coord_system = 'ALS'
                    elif 'EEGLAB' == content.get('EEGCoordinateSystemDescription', ''):
                        # some datasets with EEGLAB-style coordinates use this field instead
                        # and have other systems in the EEGCoordinateSystem field (e.g., 'CTF')
                        coord_system = 'ALS'
                    else:
                        coord_system = 'RAS'
                    coord_units = content.get(
                        'EEGCoordinateUnits', 'guess'
                    ).lower()  # default to 'guess' if not specified
                    if coord_units == 'n/a':
                        coord_units = 'guess'
                    break

            # guess the coordinate units if not specified
            coords = np.stack((elecs['x'].to_numpy(), elecs['y'].to_numpy(), elecs['z'].to_numpy()), axis=1)

            guess_units = coord_units == 'guess'

            if guess_units:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    max_coord = np.nanmax(np.abs(coords))
                if not np.isnan(max_coord):
                    if max_coord < 0.2:
                        coord_units = 'm'
                    elif max_coord < 2:
                        coord_units = 'cm'
                    elif max_coord < 20:
                        coord_units = 'mm'
                    else:
                        coord_units = ''
                else:
                    coord_units = ''
                if verbose:
                    logger.info(f"  inferred coordinate units to be in {coord_units!r}...")

            report['OriginalCoordUnits'] = coord_units
            report['CoordUnitsWereGuessed'] = guess_units

            if coord_units == '':
                warning(
                    f"Coordinate units for {fo.filename} could not be inferred "
                    f"or were invalid; not overriding channel locations."
                )
            else:
                if EEG['chaninfo']['nosedir'] != '+X':
                    warning(
                        f"Converting to the coordinate system {coord_system} of "
                        f"the EEG data file is not supported by this importer. "
                        f"Setting to +X and clearing existing coordinates."
                    )
                    # override nosedir and wipe existing chanlocs, if any
                    EEG['chaninfo']['nosedir'] = '+X'  # set to +X for AJS coordinate system
                    for ch in EEG['chanlocs']:
                        clear_chanloc(ch, numeric_null)

                # convert to mm (EEGLAB's internal unit)
                coords = coords_to_mm(coords, coord_units)
                # convert to ALS if needed
                if coord_system == 'RAS':
                    coords = coords_RAS_to_ALS(coords)
                elif coord_system != 'ALS':
                    raise ValueError(
                        f"Unsupported coordinate system {coord_system!r} "
                        f"in BIDS file {fo.filename}. Supported systems are "
                        f"ALS and RAS."
                    )

                sph_theta, sph_phi, sph_radius, polar_theta, polar_radius = coords_ALS_to_angular(coords)

                # now read in the electrode locations
                notfound = []
                num_updated = 0
                for k, ch in enumerate(elecs.iloc):
                    lab = ch['name']
                    if lab not in orig_labels:
                        notfound.append(lab)
                        continue
                    idx = orig_labels.index(lab)

                    # assign the coordinates (note we always assume AJS)
                    xyz = coords[k]

                    # update the channel record
                    rec = EEG['chanlocs'][idx]
                    if np.any(np.isnan(xyz)):
                        continue  # invalid, nothing to do
                    num_updated += 1
                    rec['X'] = xyz[0]
                    rec['Y'] = xyz[1]
                    rec['Z'] = xyz[2]
                    # also regenerate the angular coordinates
                    rec['sph_theta'] = sph_theta[k]
                    rec['sph_phi'] = sph_phi[k]
                    rec['sph_radius'] = sph_radius[k]
                    rec['theta'] = polar_theta[k]
                    rec['radius'] = polar_radius[k]
                if notfound:
                    warning(
                        f"Electrodes {','.join(notfound)} from BIDS file {fo.filename} "
                        f"not found in EEG data structure; skipping."
                    )
                if num_updated:
                    logger.info(
                        f"Updated {num_updated} channel locations from BIDS file {fo.filename} "
                        f"into the EEG data structure."
                    )
                    report['NumUpdatedChanlocs'] = num_updated
                    report['NotfoundChanlocs'] = notfound

    if bidsevent:
        import bids

        layout: bids.BIDSLayout = layout_for_fpath(filename)

        # get the query to find the associated events file
        query_entities = query_for_adjacent_fpath(filename, suffix='events', extension='.tsv')
        try:
            # retrieve the list of all such files
            events_file_list = layout.get(**query_entities, return_type='object')
            for fo in events_file_list:
                import pandas as pd

                if verbose:
                    logger.info(f"  applying BIDS events from {fo.filename}...")
                report['EventsFrom'] = os.path.relpath(fo.path, root)
                # read in the file contents
                events: pd.DataFrame = fo.get_df()

                try:
                    # opportunistically look for the 'sample' column, which may be present in some files
                    # seen in the wild
                    ev_lats = events['sample'].to_numpy()
                    if np.all(np.isnan(ev_lats)):
                        raise ValueError(f"sample column in {fo.filename} is all NaN; falling back to onsets.")
                    ev_lats = ev_lats.astype(int)
                    report['EventTimingSource'] = 'sample'
                except (KeyError, ValueError):
                    # otherwise get it from the onsets, which is expected to be always present
                    try:
                        onsets = events['onset'].to_numpy(dtype=float)
                        if np.all(np.isnan(onsets)):
                            raise ValueError(f"onset column in {fo.filename} is all NaN; cannot proceed.")
                        report['EventTimingSource'] = 'onset'
                        ev_lats = np.searchsorted(times_sec, onsets)
                    except (KeyError, ValueError):
                        raise ValueError(
                            f"Your BIDS file {fo.filename} does not contain "
                            f"the required 'onset' column for events and therefore "
                            f"does not conform to the BIDS standard; to fall back "
                            f"to the events present in the EEG file itself (if any), "
                            f"pass the bidsevent=False option "
                            f"when using pop_load_frombids, or equivalently "
                            f"ApplyEvents=False when using  bids_preproc()."
                        )
                # convert to 1-based indexes for MATLAB compat
                ev_lats = ev_lats + 1

                try:
                    durations = events['duration'].to_numpy(dtype=float).copy()
                    durations[np.isnan(durations)] = 0.0  # replace NaNs with zero
                except KeyError:
                    # fall back to zero duration
                    durations = np.zeros_like(onsets, dtype=float)
                # convert to EEGLAB's sample-based durations
                ev_durs = round_mat(np.maximum(1, Fs * durations)).astype(int)

                # set of column names that we've already carried over into the
                # event data structure
                used_columns = list(event_timing_columns)

                if eventtype:
                    # restrict to the specified event type only
                    probe_columns = [eventtype]
                else:
                    probe_columns = event_type_columns

                # read out the event types and/or codes

                for candidate_column in probe_columns:
                    try:
                        # preferred column for the event type
                        ev_types = events[candidate_column].to_numpy()
                        report['EventSourceColumn'] = candidate_column
                        EEG['etc']['event_column'] = candidate_column
                        used_columns.append(candidate_column)
                        break
                    except KeyError:
                        # not found
                        continue
                else:
                    warning(
                        f"Your BIDS file {fo.filename} does not appear to contain "
                        f"a column coding for the event type ({','.join(event_type_columns)}), "
                        f"importing as ''. To avoid importing these dummy events and use only"
                        f"the events in the EEG file itself (if any), pass the "
                        f"bidsevent=False option when using pop_load_frombids, "
                        f"or equivalently ApplyEvents=False when using bids_preproc()."
                    )
                    ev_types = np.full_like(ev_lats, '', dtype=object)

                ev_types = [typ or ('boundary' if typ == 'New Segment' else '') for typ in ev_types]

                # extract extra columns to include in the event data structure
                # this does not in
                extra_columns = sorted(set(events.columns) - set(used_columns))
                ev_extra = {col: events[col].to_numpy() for col in extra_columns}

                # drop trivial (all-nan) columns
                for col in list(ev_extra):
                    try:
                        if np.all(np.isnan(ev_extra[col])):
                            del ev_extra[col]
                    except Exception:
                        pass

                # filter out any events that are already in the EEG data structure itself
                # noinspection PyBroadException
                try:
                    if bidsevent in ('replace', True):
                        EEG['event'] = np.array([], dtype=object)  # clear existing events
                        keep = np.ones_like(ev_types, dtype=bool)
                    elif bidsevent == 'merge':
                        if len(EEG['event']):
                            orig_lats = [e['latency'] for e in EEG['event']]
                            indexes = np.searchsorted(orig_lats, ev_lats)
                            orig_types = [ev['type'] for ev in EEG['event'][indexes]]
                            keep = [o != e for o, e in zip(orig_types, ev_types)]
                        else:
                            keep = np.ones_like(ev_types, dtype=bool)
                    elif bidsevent == 'append':
                        keep = np.ones_like(ev_types, dtype=bool)
                    else:
                        raise ValueError(
                            f"Invalid value for bidsevent: {bidsevent}. "
                            f"Expected one of 'replace', 'merge', 'append', or False/None."
                        )

                    report["NumEventsFromBids"] = int(np.sum(keep))

                    # append the new events to the EEG structure
                    if count := np.sum(keep):
                        # build an events structure (SoA form)
                        events_soa = {
                            'latency': ev_lats,
                            'duration': ev_durs,
                            'type': ev_types,
                            'urevent': np.zeros_like(ev_lats),
                        }
                        # append extra event columns
                        events_soa.update(ev_extra)

                        # convert from structure-of-arrays to array-of-structures and filter by keep
                        new_events = [
                            {key: values[i] for key, values in events_soa.items()} for i, kp in enumerate(keep) if kp
                        ]

                        EEG_events = np.asarray(EEG['event'], dtype=object).tolist()

                        # append any missing fields to existing events with null values
                        for col in ev_extra:
                            for ev in EEG_events:
                                if col not in ev:
                                    ev[col] = numeric_null

                        EEG_events = np.asarray(EEG['event'], dtype=object).tolist() + new_events
                        EEG['event'] = np.array(EEG_events, dtype=object)

                        # re-sort events by latency
                        lats = [ev['latency'] for ev in EEG['event']]
                        EEG['event'] = EEG['event'][np.argsort(lats)]

                        # rewrite the urevent index since it'll have gotten scrambled
                        for i, ev in enumerate(EEG['event']):
                            ev['urevent'] = i

                        # rewrite urevent itself
                        EEG['urevent'] = copy.deepcopy(EEG['event'])

                        logger.info(
                            f"Merged {count} events from the BIDS events file {fo.filename} "
                            f"into the EEG file {basename}."
                        )

                    report["NumEventsFromEEGFile"] = len(EEG['event']) - int(np.sum(keep))

                except ExceptionUnlessDebug:
                    logger.exception(
                        f"Failed to deduplicate events between the EEG file {basename} "
                        f"and the BIDS events file {fo.filename}; keeping all events."
                    )
        except ExceptionUnlessDebug:
            logger.exception(
                f"Failed to load BIDS events file for {filename}. Only the events "
                f"in the EEG file itself will be retained."
            )

    coords = chanlocs_to_coords(EEG['chanlocs'])
    have_coords = not np.all(np.isnan(coords))
    if infer_locations is None:
        infer_locations = not have_coords  # only if no coordinates are present

    apply_montage_inference(
        EEG,
        infer_locations,
        numeric_null=numeric_null,
        report=report,
        warning=warning,
        error=error,
    )

    EEG = eeg_checkset(EEG)
    try:
        from eegprep import eeg_checkchanlocs

        EEG = eeg_checkchanlocs(EEG)
    except ImportError:
        logger.info("eeg_checkchanlocs not available, skipping channel location check.")

    # Assign channel types based on channel labels (matching MATLAB's eeg_getchantype behavior)
    # Standard 10-20 channel names that should be classified as EEG
    # From EEGLAB's Standard-10-20-Cap81.locs (exact copy)
    standard_eeg_channels = [
        'Fp1',
        'Fpz',
        'Fp2',
        'Nz',
        'AF9',
        'AF7',
        'AF3',
        'AFz',
        'AF4',
        'AF8',
        'AF10',
        'F9',
        'F7',
        'F5',
        'F3',
        'F1',
        'Fz',
        'F2',
        'F4',
        'F6',
        'F8',
        'F10',
        'FT9',
        'FT7',
        'FC5',
        'FC3',
        'FC1',
        'FCz',
        'FC2',
        'FC4',
        'FC6',
        'FT8',
        'FT10',
        'T9',
        'T7',
        'C5',
        'C3',
        'C1',
        'Cz',
        'C2',
        'C4',
        'C6',
        'T8',
        'T10',
        'TP9',
        'TP7',
        'CP5',
        'CP3',
        'CP1',
        'CPz',
        'CP2',
        'CP4',
        'CP6',
        'TP8',
        'TP10',
        'P9',
        'P7',
        'P5',
        'P3',
        'P1',
        'Pz',
        'P2',
        'P4',
        'P6',
        'P8',
        'P10',
        'PO9',
        'PO7',
        'PO3',
        'POz',
        'PO4',
        'PO8',
        'PO10',
        'O1',
        'Oz',
        'O2',
        'O9',
        'O10',
        'CB1',
        'CB2',
        'Iz',
    ]
    standard_eeg_channels_upper = [ch.upper() for ch in standard_eeg_channels]

    # Channel type keywords (from BIDS specification)
    type_keywords = ['EEG', 'MEG', 'MEGREF', 'SEEG', 'EMG', 'EOG', 'ECG', 'EKG', 'TRIG', 'GSR', 'PPG', 'MISC']

    for i, ch in enumerate(EEG['chanlocs']):
        label = ch.get('labels', '')
        current_type = ch.get('type')

        # Skip if type is already properly assigned (not nan, not 'n/a', not empty)
        if isinstance(current_type, str) and current_type and current_type.lower() not in ['n/a', 'nan']:
            continue

        # Check if label contains any type keyword
        assigned = False
        for keyword in type_keywords:
            if keyword.lower() in label.lower():
                EEG['chanlocs'][i]['type'] = keyword
                assigned = True
                break

        # If no keyword match, check if it's a standard 10-20 channel name
        if not assigned and label.upper() in standard_eeg_channels_upper:
            EEG['chanlocs'][i]['type'] = 'EEG'

    if return_report:
        return EEG, report
    else:
        return EEG
