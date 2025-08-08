
import os
import copy
from typing import Dict, Any, Tuple, List
import logging
import warnings
import contextlib
from .utils.bids import layout_for_fpath, layout_get_lenient, query_for_adjacent_fpath
from .utils import ExceptionUnlessDebug

import numpy as np

logger = logging.getLogger(__name__)


# list of candidate column names for event types in BIDS events files, in order of preference.
event_type_columns = ['trial_type', 'type', 'event_type', 'HED', 'value', 'code']


def pop_load_frombids(
        filename: str,
        *,
        apply_bids_metadata: bool = True,
        apply_bids_channels: bool = True,
        apply_bids_events: bool = False,
        dtype: np.dtype = np.float32,
        numeric_null: Any = np.array([]),
        verbose: bool = True,
) -> Dict[str, Any]:
    """Load an EEG data file of a supported format from a BIDS dataset.

    Supported formats are EDF, BrainVision, EEGLAB SET, BDF.

    Args:
        filename: Path to the EEG data file in a BIDS dataset.
        apply_bids_metadata: Whether to override any metadata in the EEG file with
          metadata from BIDS.
        apply_bids_channels: Whether to override any channel information (incl. locations)
          in the EEG file with channel information from BIDS.
        apply_bids_events: Whether to override any event data in the EEG file with
          event data from BIDS.
        dtype: The data type to use for the EEG data.
        numeric_null: The value to use for empty numeric fields in the EEG data.
          * the default is np.array([]) for MATLAB/pop_loadset compatibility

    Returns:
        EEG: A dictionary containing the EEG data and metadata.

    """

    path, ext = os.path.splitext(filename)
    ext = ext.lower()

    if verbose:
        logger.info(f"Loading EEG data from {filename}...")
    if ext == '.set':
        from . import pop_loadset
        EEG = pop_loadset(filename)
        EEG['data'] = EEG['data'].astype(dtype)
    elif ext in ['.edf', '.bdf', '.vhdr']:
        if ext == '.vhdr':
            from neo.rawio.brainvisionrawio import BrainVisionRawIO as NeoIO
        elif ext in ['.edf', '.bdf']:
            from neo.rawio.edfrawio import EDFRawIO as NeoIO
        else:
            # if you're getting this, there's an elif statement missing here for one of
            # the formats allowed above
            raise ValueError(f"Unexpected file format: {ext}. Please add support for this "
                             f"format if needed.")
        # load from NEO
        io = NeoIO(filename)
        io.parse_header()
        if (nStreams := io.signal_streams_count()) > 1:
            logger.warning(f"The raw data file {filename} appears to contain "
                           f"more than one stream; using only the first stream.")
        elif not nStreams:
            raise ValueError(f"The raw data file {filename} does not contain any data.")
        if (nBlocks := io.block_count()) > 1:
            logger.warning(f"The raw data file {filename} appears to contain "
                           f"more than one recording; this is not meaningful "
                           f"in a BIDS context; using only the first block.")
        elif not nBlocks:
            raise ValueError(f"The raw data file {filename} does not contain any data.")
        if (nSegments := io.segment_count(0)) > 1:
            raise NotImplementedError(f"The raw data file {filename} appears to contain "
                                      f"more than one segment; This importer currently "
                                      f"only supports continuous EEG data.")
        elif not nSegments:
            raise ValueError(f"The raw data file {filename} does not contain any data.")

        nChannels = io.signal_channels_count(0)
        nSamples = io.get_signal_size(0, 0, 0)
        chnIdxs = list(range(nChannels))

        if verbose:
            logger.info("  retrieving EEG data from file...")
        data_T = io.get_analogsignal_chunk(block_index=0, seg_index=0,
                                           channel_indexes=chnIdxs,
                                           i_start=None, i_stop=None)
        data_T = io.rescale_signal_raw_to_float(data_T, dtype=dtype,
                                                channel_indexes=chnIdxs)

        # data time codes
        Fs = io.get_signal_sampling_rate(0)
        t0 = io.get_signal_t_start(block_index=0, seg_index=0, stream_index=0)
        t0 += getattr(io, '_global_time', 0.0)  # default to 0 if not set
        times = t0 + np.arange(0, nSamples, dtype=float) / Fs

        # construct the chanlocs data structure
        chns = io.header['signal_channels']
        # get the units for all channels
        try:
            units = chns['units'].tolist()
        except KeyError:
            units = ['uV']*nChannels
        uq_unit = np.unique(units)
        if len(uq_unit) == 1 and uq_unit[0] not in ('uV', 'microvolts'):
            logger.warning(f"Your channel unit does not appear to be in microvolts (uV) "
                           f"but is documented instead as {uq_unit[0]}. EEG scale might be incorrect. ")

        labels = chns['name'].tolist()

        # other available per-channel fields from neo:
        # - id
        # - sampling_rate (assumed to be uniform across all channels)
        # - dtype
        # - gain  (accounted for in rescaling)
        # - offset (accounted for in rescaling)
        # - stream_id
        # - buffer_id

        # preinitialize data structure
        chanlocs = np.asarray([
            {
                'labels': lab,
                'sph_radius': numeric_null,
                'sph_theta': numeric_null,
                'sph_phi': numeric_null,
                'theta': numeric_null,
                'radius': numeric_null,
                'X': numeric_null,
                'Y': numeric_null,
                'Z': numeric_null,
                'type': 'EEG',
                'ref': numeric_null,
                'urchan': numeric_null
            } for lab in labels])

        # try to read out channel coordinates from side-channel info, if any
        if ext == '.vhdr':
            if verbose:
                logger.info("  parsing VHDR-specific channel locations...")
            try:
                annots = io.raw_annotations['blocks'][0]['segments'][0]['signals'][0]['__array_annotations__']
                sph_radius, theta, phi = annots['coordinates_0'], annots['coordinates_1'], annots['coordinates_2']
                valid = (sph_radius!=0) | (theta!=0) | (phi!=0)
                sph_theta = phi - 90 * np.sign(theta)
                sph_phi = -np.abs(theta) + 90
            except KeyError:
                logger.warning(f"Channel coordinates not found in {filename}. "
                               f"Using default values for channel locations.")
                valid = np.zeros(nChannels, dtype=bool)
        elif ext in ['.edf', '.bdf']:
            # EDF/BDF files do not have channel coordinates, so we use default values
            valid = np.zeros(nChannels, dtype=bool)
        else:
            raise ValueError(f"Unsupported file format for channel coordinates extraction: {ext}. "
                             f"Supported formats are .edf, .bdf, .vhdr.")

        if np.any(valid):
            if verbose:
                logger.info("  applying channel locations from EEG file...")
            # set the channel locations to the extent that we have them
            for loc, val, sph_r, sph_p, sph_t in zip(chanlocs, valid, sph_radius, sph_phi, sph_theta):
                if val:
                    # write coordinates in
                    loc['sph_radius'] = sph_r
                    loc['sph_theta'] = sph_t
                    loc['sph_phi'] = sph_p
                    # also derive topo coords (sph2topo)
                    az = sph_p
                    horiz = sph_t
                    angle = -horiz
                    radius = 0.5 - az/180
                    loc['theta'] = angle
                    loc['radius'] = radius
                    # and derive cartesian coordinates (sph2cart)
                    az = np.deg2rad(sph_t)
                    elev = np.deg2rad(sph_p)
                    z = sph_r * np.sin(elev)
                    x = sph_r * np.cos(elev) * np.cos(az)
                    y = sph_r * np.cos(elev) * np.sin(az)
                    loc['X'] = x
                    loc['Y'] = y
                    loc['Z'] = z

        # construct the events data structure
        if (nEvtChns := io.event_channels_count()) > 0:
            if verbose:
                logger.info("  reading in event data from EEG file...")
            ev_all_times = []
            ev_all_durs = []
            ev_all_channels = []
            ev_all_data = []
            # All channels containing events get collapsed into a single axis of instances.
            # The instance 'label' contains the original channel name.
            # The instance 'data' contains the original event marker.
            for ev_ch_ix in range(nEvtChns):
                ev_times, ev_durs, ev_labels = io.get_event_timestamps(
                    block_index=0,
                    seg_index=0,
                    event_channel_index=ev_ch_ix,
                    t_start=None,
                    t_stop=None,
                    # (no other args)
                )
                ev_all_times.extend(io.rescale_event_timestamp(ev_times))
                if ev_durs is not None:
                    ev_all_durs.extend(ev_durs)
                else:
                    ev_all_durs.extend([1] * len(ev_times))
                ev_all_channels.extend(np.repeat(io.header['event_channels'][ev_ch_ix]['name'], len(ev_times)))
                ev_all_data.extend(ev_labels)
            # apply heuristics to deduce the event type
            if ext == '.vhdr':
                # BrainVision has the event name in the data, but when that's empty,
                # we use the channel name as the event type.
                ev_types = ev_all_data
                ev_codes = ev_all_channels
            elif ext in ['.edf', '.bdf']:
                ev_types = [str(d) for d in ev_all_data]
                ev_codes = [str(chn) for chn in ev_all_channels]
            else:
                # if you get this you need to add support for this file format here
                raise ValueError(f"Unsupported file format for event extraction: {ext}. "
                                 f"Supported formats are .edf, .bdf, .vhdr.")
            ev_lats = np.searchsorted(times, ev_all_times)  # +1 for MATLAB format compatibility (1-based index)
            ev_durs = np.array(ev_all_durs, dtype=float)
            ev_urevts = 1 + np.arange(len(ev_all_times))
            events = np.array([
                {
                    'duration': dur,
                    'latency': lat,
                    'type': typ or ('boundary' if code == 'New Segment' else ''),
                    'code': code,
                    'urevent': chn,
                } for dur, lat, typ, code, chn in
                zip(ev_durs, ev_lats, ev_types, ev_codes, ev_urevts)])
        else:
            events = numeric_null

        # this isn't really encoded in Neo's data structure, nor does pop_loadbv() seem
        # to read it out, even though .vhdr CAN have it annotated (either the [Comments] section
        # of the channel infos in the .vhdr file, or separately in each channel under [Channel Infos]
        reference = 'unknown'

        basename = os.path.basename(filename)
        EEG = {
            'setname': '',
            'filename': basename,
            'filepath': os.path.dirname(filename),
            # these will be set from BIDS
            'subject': '',
            'group': '',
            'condition': '',
            'session': numeric_null,
            'comments': '',
            # raw data array
            'nbchan': nChannels,
            'trials': 1,  # assuming single trial for raw EEG datain
            'pnts': nSamples,
            'srate': Fs,
            'xmin': times[0],
            'xmax': times[-1],
            'times': times*1000,  # in ms
            'data': data_T.T,
            # ICA data structures
            'icaact': numeric_null,
            'icawinv': numeric_null,
            'icasphere': numeric_null,
            'icaweights': numeric_null,
            'icachansind': numeric_null,
            # channel info
            'chanlocs': chanlocs,
            'urchanlocs': numeric_null,
            'chaninfo': {
                'plotrad': numeric_null,
                'shrink': numeric_null,
                'nosedir': '+X',
                'nodatchans': numeric_null,
                'icachansind': numeric_null,
            },
            'ref': reference,
            # event data structures
            'event': events,
            'urevent': copy.deepcopy(events),
            'eventdescription': [],
            # epoch info
            'epoch': numeric_null,
            'epochdescription': [],
            # rejection info (note: could pre-populate)
            'reject': {},
            'stats': {},
            # spectral data (not used)
            'specdata': numeric_null,
            'specicaact': numeric_null,
            # spline fil
            'splinefile': '',
            'icasplinefile': '',
            # DIPFIT info
            'dipfit': numeric_null,
            # history info
            'history': '',
            'saved': 'justloaded',
            # additional metadata
            'etc': {},
            'run': numeric_null
        }
    elif ext in ['.fdt', '.vmrk', '.eeg']:
        raise ValueError(f"pop_load_frombids should be called with the main data file, "
                         f"but was called on a sidecar file: {filename}.")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats "
                         f"are .set, .edf, .bdf, .vhdr.")

    if apply_bids_metadata:
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

    if apply_bids_channels:
        import bids
        layout: bids.BIDSLayout = layout_for_fpath(filename)

        # check for presence of a _channels.tsv file
        query_entities = {
            **layout.parse_file_entities(filename),
            'suffix': 'channels',
            'extension': '.tsv'
        }
        # retrieve the list of all such files
        channel_file_list = layout_get_lenient(
            layout,
            **query_entities,
            return_type='object',
            tolerate_missing=('task', 'run')
        )
        if len(channel_file_list) > 1:
            logger.warning(f"Found multiple BIDS channel files for {filename}: "
                           f"{', '.join([fo.filename for fo in channel_file_list])}. "
                           f"Using the first one only.")
        for fo in channel_file_list:
            import pandas as pd
            if verbose:
                logger.info(f"  applying BIDS channel locations from {fo.filename}...")
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
                logger.warning(f"Channels {','.join(notfound)} from BIDS file {fo.filename} "
                               f"not found in EEG data structure; skipping.")
            if notype:
                logger.warning(f"Cchannels in BIDS file {fo.filename} do not have a 'type' "
                               f"column; not overriding.")

            break

        # check for presence of an _electrodes.tsv file
        query_entities = query_for_adjacent_fpath(
            filename, suffix='electrodes', extension='.tsv')
        # retrieve the list of all such files
        elec_file_list = layout_get_lenient(
            layout,
            **query_entities,
            return_type='object',
            tolerate_missing=('task', 'run'),
        )
        if len(elec_file_list) > 1:
            logger.warning(f"Found multiple BIDS electrode files for {filename}: "
                           f"{', '.join([fo.filename for fo in elec_file_list])}. "
                           f"Using the first one only.")
        for fo in elec_file_list:
            import pandas as pd
            if verbose:
                logger.info(f"  applying BIDS electrode locations from {fo.filename}...")
            # read in the file contents
            elecs: pd.DataFrame = fo.get_df()

            # check for the presence of a coordsystem file
            query_entities = query_for_adjacent_fpath(
                fo.path, suffix='coordsystem', extension='.json')
            coordsystem_file_list = layout_get_lenient(
                layout,
                **query_entities,
                return_type='object',
                tolerate_missing=('task', 'run', 'space'),
            )
            if len(coordsystem_file_list) > 1:
                logger.warning(f"Found multiple BIDS coordsystem files for {fo.filename}: "
                               f"{', '.join([fo.filename for fo in coordsystem_file_list])}. "
                               f"Using the first one only.")
            elif not coordsystem_file_list:
                # if it's a .set study, then we assume ALS for the chanlocs, otherwise RAS
                coord_system = 'ALS' if ext == '.set' else 'RAS'
                coord_units = 'guess'
                logger.warning(f"Found no BIDS coordsystem files for {fo.filename}; your "
                               f"dataset is not fully BIDS-compliant. Assuming coordinate "
                               f"system {coord_system!r} and guessing units from the data.")
            else:
                for coordsystem_fo in coordsystem_file_list:
                    if verbose:
                        logger.info(f"  applying BIDS coordsystem from {coordsystem_fo.filename}...")
                    # read in the file contents
                    content: Dict[str, Any] = coordsystem_fo.get_dict()
                    coord_system = content.get('EEGCoordinateSystem', 'RAS')  # default to RAS if not specified
                    if 'EEGLAB' in coord_system.upper():
                        # as per BIDS docs, EEGLAB is the only one that's expressly not RAS
                        coord_system = 'ALS'
                    else:
                        coord_system = 'RAS'
                    coord_units = content.get('EEGCoordinateUnits', 'guess').lower()  # default to 'guess' if not specified
                    if coord_units == 'n/a':
                        coord_units = 'guess'
                    break

            # guess the coordinate units if not specified
            if coord_units == 'guess':
                coords = np.stack((elecs['x'].to_numpy(), elecs['y'].to_numpy(), elecs['z'].to_numpy()), axis=1)
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

            if coord_units == '':
                logger.warning(f"Coordinate units for {fo.filename} could not be inferred "
                               f"or were invalid; not overriding channel locations.")
            else:
                if EEG['chaninfo']['nosedir'] != '+X':
                    logger.warning(
                        f"Converting to the coordinate system {coord_system} of "
                        f"the EEG data file is not supported by this importer. "
                        f"Setting to +X and clearing existing coordinates.")
                    # override nosedir and wipe existing chanlocs, if any
                    EEG['chaninfo']['nosedir'] = '+X'  # set to +X for AJS coordinate system
                    for ch in EEG['chanlocs']:
                        ch['sph_radius'] = numeric_null
                        ch['sph_theta'] = numeric_null
                        ch['sph_phi'] = numeric_null
                        ch['theta'] = numeric_null
                        ch['radius'] = numeric_null
                        ch['X'] = numeric_null
                        ch['Y'] = numeric_null
                        ch['Z'] = numeric_null

                # convert to mm (EEGLAB's internal unit)
                if coord_units == 'mm':
                    pass
                elif coord_units == 'cm':
                    coords *= 10.0
                elif coord_units == 'm':
                    coords *= 1000.0

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
                    if coord_system == 'ALS':
                        # applies as-is
                        x = rec['X'] = xyz[0]
                        y = rec['Y'] = xyz[1]
                        z = rec['Z'] = xyz[2]
                    elif coord_system == 'RAS':
                        # map from RAS to ALS
                        x = rec['X'] = xyz[1]  # A is second position
                        y = rec['Y'] = -xyz[0] # L is first position, but inverted
                        z = rec['Z'] = xyz[2]  # S is third position, as-is
                    else:
                        raise ValueError(f"Unsupported coordinate system {coord_system!r} "
                                         f"in BIDS file {fo.filename}. Supported systems are "
                                         f"ALS and RAS.")
                    # also regenerate the spherical coordinates (cart2sph)
                    hypotxy = np.hypot(x, y)
                    theta = np.arctan2(y, x)
                    phi = np.arctan2(z, hypotxy)
                    radius = np.hypot(hypotxy, z)
                    rec['sph_theta'] = theta / np.pi * 180
                    rec['sph_phi'] = phi / np.pi * 180
                    rec['sph_radius'] = radius
                    # and the 2d topographic coordinates (sph2topo)
                    rec['theta'] = -rec['sph_theta']
                    rec['radius'] = 0.5 - rec['sph_phi']/180

                if notfound:
                    logger.warning(f"Electrodes {','.join(notfound)} from BIDS file {fo.filename} "
                                   f"not found in EEG data structure; skipping.")
                if num_updated:
                    logger.info(f"Updated {num_updated} channel locations from BIDS file {fo.filename} "
                                f"into the EEG data structure.")

    if apply_bids_events:
        import bids
        layout: bids.BIDSLayout = layout_for_fpath(filename)

        # get the query to find the associated events file
        query_entities = query_for_adjacent_fpath(
            filename, suffix='events', extension='.tsv')
        try:
            # retrieve the list of all such files
            events_file_list = layout.get(**query_entities, return_type='object')
            for fo in events_file_list:
                import pandas as pd
                if verbose:
                    logger.info(f"  applying BIDS events from {fo.filename}...")
                # read in the file contents
                events: pd.DataFrame = fo.get_df()
                try:
                    # opportunistically look for the 'sample' column, which may be present in some files
                    # seen in the wild
                    ev_lats = events['sample'].to_numpy(dtype=int)
                except KeyError:
                    # otherwise get it from the onsets, which is expected to be always present
                    try:
                        onsets = events['onset'].to_numpy(dtype=float)
                        ev_lats = np.searchsorted(times, onsets)
                    except KeyError as e:
                        raise ValueError(f"Your BIDS file {fo.filename} does not contain "
                                         f"the required 'onset' column for events and therefore "
                                         f"does not conform to the BIDS standard; to fall back "
                                         f"to the events present in the EEG file itself (if any), "
                                         f"pass the apply_bids_events=False option "
                                         f"when using pop_load_frombids, or equivalently "
                                         f"ApplyEvents=False when using  bids_preproc().")

                try:
                    durations = events['duration'].to_numpy(dtype=float)
                    durations[np.isnan(durations)] = 0.0  # replace NaNs with zero
                except KeyError:
                    # fall back to zero duration
                    durations = np.zeros_like(onsets, dtype=float)
                # convert to EEGLAB's sample-based durations
                ev_durs = np.round(np.maximum(1, Fs*durations)).astype(int)

                # read out the event types and/or codes
                for candidate_column in event_type_columns:
                    try:
                        # preferred column for the event type
                        ev_types = events[candidate_column].to_numpy()
                        break
                    except KeyError:
                        # not found
                        continue
                else:
                    logger.warning(f"Your BIDS file {fo.filename} does not appear to contain "
                                   f"a column coding for the event type ({','.join(event_type_columns)}), "
                                   f"importing as ''. To avoid importing these dummy events and use only"
                                   f"the events in the EEG file itself (if any), pass the "
                                   f"apply_bids_events=False option when using pop_load_frombids, "
                                   f"or equivalently ApplyEvents=False when using bids_preproc().")
                    ev_types = np.full_like(ev_lats, '', dtype=object)

                ev_types = [typ or ('boundary' if typ == 'New Segment' else '') for typ in ev_types]

                # filter out any events that are already in the EEG data structure itself
                # noinspection PyBroadException
                try:
                    if len(EEG['event']):
                        orig_lats = [e['latency'] for e in EEG['event']]
                        indexes = np.searchsorted(orig_lats, ev_lats)
                        orig_types = [ev['type'] for ev in EEG['event'][indexes]]
                        keep = [o != e for o,e in zip(orig_types, ev_types)]
                    else:
                        keep = np.ones_like(ev_types, dtype=bool)

                    # append the new events to the EEG structure
                    if count := np.sum(keep):
                        EEG_events = EEG['event'].tolist()
                        for kp, lat, dur, typ in zip(keep, ev_lats, ev_durs, ev_types):
                            if kp:
                                EEG_events.append({
                                    'latency': lat,
                                    'duration': dur,
                                    'type': typ,
                                    'urevent': 0  # urevent is 1-based index
                                })

                        EEG['event'] = np.array(EEG_events, dtype=object)

                        # resort events by latency
                        lats = [ev['latency'] for ev in EEG['event']]
                        EEG['event'] = EEG['event'][np.argsort(lats)]

                        # rewrite the urevent index since it'll have gotten scrambled
                        for i, ev in enumerate(EEG['event']):
                            ev['urevent'] = i + 1

                        # rewrite urevent itself
                        EEG['urevent'] = copy.deepcopy(EEG['event'])

                        logger.info(f"Merged {count} events from the BIDS events file {fo.filename} " 
                                    f"into the EEG file {basename}.")
                except ExceptionUnlessDebug:
                    logger.exception(f"Failed to deduplicate events between the EEG file {basename} "
                                     f"and the BIDS events file {fo.filename}; keeping all events.")
        except ExceptionUnlessDebug:
            logger.exception(f"Failed to load BIDS events file for {filename}. Only the events "
                             f"in the EEG file itself will be retained.")

    return EEG
