import os
from typing import Sequence, Dict, Any

__all__ = ['root_for_fpath', 'gen_derived_fpath', 'layout_for_fpath', 'layout_get_lenient', 'query_for_adjacent_fpath']


# cache of BIDS layout objects, by root path
layout_cache = {}


def root_for_fpath(fn: str) -> str:
    """Get the root directory of a BIDS dataset from a file path within it."""
    normpath = os.path.normpath(fn)
    pathparts = normpath.split(os.sep)[:-1]
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
        toplevel: str = 'clean_artifacts',
        keyword: str = 'cleaned'
) -> str:
    """Generate a file path for a derived EEG file in a BIDS dataset.

    Args:
        raw_fn: original raw filename
        toplevel: top-level directory for derived files (e.g., 'clean_artifacts')
        keyword: keyword to splice into the filename (e.g., 'cleaned')

    """
    fn = raw_fn
    root = root_for_fpath(fn)
    basedir = os.path.dirname(fn)
    root_relative = os.path.relpath(basedir, root)
    relative_dirname = os.path.dirname(root_relative)
    raw_fname = os.path.basename(fn)
    raw_fname_parts = raw_fname.split('_')
    new_fname = '_'.join(raw_fname_parts[:-1]) + f'_{keyword}_' + raw_fname_parts[-1]
    new_fprefix, old_fext = os.path.splitext(new_fname)
    new_fext = '.set'  # derived data is always in .set format
    out_path = f"{root}/derivatives/{toplevel}/{relative_dirname}/{new_fprefix}{new_fext}"
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
