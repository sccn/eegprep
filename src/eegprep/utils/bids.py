import os
from typing import Sequence, Dict, Any, Optional


__all__ = ['root_for_fpath', 'gen_derived_fpath', 'layout_for_fpath', 'layout_get_lenient', 'query_for_adjacent_fpath']


# cache of BIDS layout objects, by root path
layout_cache = {}

# incomplete list of bids suffixes that may occur in an EEG/multimodal study
# (should not be used for verification; authoritative list at:
# https://github.com/bids-standard/bids-specification/blob/master/src/schema/objects/suffixes.yaml)
bids_suffixes = ('beh', 'channels', 'coordsystem', 'descriptions', 'desc', 'eeg', 'events', 'electrodes',
                 'headshape', 'markers', 'meg', 'motion', 'nirs', 'optodes', 'physio', 'sessions',
                 'stim')


def root_for_fpath(fn: str) -> str:
    """Get the root directory of a BIDS dataset from a file path within it."""
    normpath = os.path.normpath(fn)
    pathparts = normpath.split(os.sep)
    # strip off the filename portion, if any
    if not os.path.isdir(normpath):
        pathparts = pathparts[:-1]
    # strip modality
    if pathparts[-1] == 'eeg':
        pathparts = pathparts[:-1]
    # strip session, if present
    if pathparts[-1].startswith('ses-'):
        pathparts = pathparts[:-1]
    # strip subject, if present
    if pathparts[-1].startswith('sub-'):
        pathparts = pathparts[:-1]
    # check if we have a 'derivatives' folder still further upstream
    if 'derivatives' in pathparts:
        bids_root = normpath[:normpath.rfind('derivatives') - 1]
    else:
        bids_root = os.sep.join(pathparts)
    return bids_root


def query_for_adjacent_fpath(
        fn: str,
        **overrides
) -> Dict[str, Any]:
    """Generate a quary dictionary (of entities) for a given file path in a BIDS dataset,
    where we selectively apply overrides to the entities."""
    layout = layout_for_fpath(fn)
    query_entities = layout.parse_file_entities(fn).copy()
    query_entities.update(overrides)
    return query_entities


def gen_derived_fpath(
        raw_fn: str,
        *,
        outputdir: str = '${root}/derivatives/clean_artifacts',
        keyword: str = '',
        suffix: Optional[str] = None,
        extension: str = '.set'
) -> str:
    """Generate a file path for a derived EEG file in a BIDS dataset.

    Args:
        raw_fn: original raw filename
        outputdir: output directory for derived files (e.g., 'derivatives/clean_artifacts')
        keyword: optional keyword tag to splice into the filename (e.g., 'desc-cleaned')
        suffix: optionally an override for the suffix (or '' to drop the existing suffix,
          if any and if it's recognized as such)
        extension: file extension for the newly generated file

    """
    fn = raw_fn
    root = root_for_fpath(fn)
    basedir = os.path.dirname(fn)
    root_relative = os.path.relpath(basedir, root)
    raw_fname, old_fext = os.path.splitext(os.path.basename(fn))
    raw_fname_parts = raw_fname.split('_')
    if keyword:
        raw_fname_parts.insert(-1, keyword)
    if suffix is not None and raw_fname_parts[-1] in bids_suffixes:
        if not suffix:
            raw_fname_parts.pop(-1)
        else:
            raw_fname_parts[-1] = suffix
    new_fprefix = '_'.join(raw_fname_parts)
    if not extension.startswith('.'):
        extension = '.' + extension
    new_fext = extension  # derived data is always in .set format
    if os.path.isabs(outputdir):
        pass  # full abs path, no further change necessary
    else:
        # substitute {root} in outputdir with the relative root path
        if '{root}' in outputdir:
            outputdir = outputdir.format(root=root)
        elif outputdir.startswith('derivatives/'):
            # if outputdir starts with 'derivatives/', we need to prepend the relative root path
            outputdir = os.path.join(root_relative, outputdir)
        elif len(outputdir.split(os.sep)) == 1:
            # single directory name, need to prepend everything else
            outputdir = os.path.join(root_relative, 'derivatives', outputdir)

    out_path = f"{outputdir}/{root_relative}/{new_fprefix}{new_fext}"
    return out_path


def layout_for_fpath(filepath: str) -> 'bids.BIDSLayout':
    """Get the applicable BIDS layout object for a given file path."""
    bids_root = root_for_fpath(filepath)
    try:
        return layout_cache[bids_root]
    except KeyError:
        import bids
        layout = bids.BIDSLayout(bids_root, validate=False)
        layout_cache[bids_root] = layout
        return layout


def layout_get_lenient(
        layout: 'bids.BIDSLayout',
        *,
        return_type: str = 'filename',
        tolerate_missing: Sequence[str] = ('task', 'run'),
        **filters,
) -> list:
    """Wrapper for layout.get() that tolerates specific missing entities, in the
    specified order of succession.

    Args:
        layout: BIDSLayout object to query.
        **kwargs: Query parameters for the layout.get() method.
        return_type: Type of return value, e.g., 'filename', 'object', etc.
          Defaults to 'filename'.
        tolerate_missing: Sequence of entity names that can be missing in the query.
          The method will progressively strip these entities from the query until
          a match is found or there are no more candidates to strip.

    Returns:
        List of return values matching the query.
    """
    query_entities = filters.copy()
    for candidate in (None,) + tuple(tolerate_missing):
        if candidate is not None and candidate in query_entities:
            del query_entities[candidate]
        results = layout.get(**query_entities, return_type=return_type)
        if results:
            return results
    # If we reach here, no results were found
    return []
