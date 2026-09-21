"""EEGPrep's browser runtime: read a NEMAR recording without installing EEGPrep.

This is the lean distribution decided in ADR 0069. eegprep stays whole and keeps every
dependency it has; this package is built to a download budget instead, because a browser
session pays for its dependencies before it sees anything.

Reads go through ``zarr.nemar.org``, which is the contract. The index also publishes
``data_base``, naming the bucket the bytes sit in today, and this package does not follow
it: see :mod:`eegprep_lean.index`.

Everything that touches the network is ``async``. That is not a style choice. Pyodide's
main thread cannot start the IO thread that synchronous Zarr requires, and the resulting
error names threads rather than Zarr or the browser, so there is no synchronous wrapper
to offer here.

    from eegprep_lean import read_index

    index = await read_index("nm000103")
    store = index.stores[0]
    url = index.level0_url(store)
"""

from .index import (
    SUPPORTED_FORMAT_VERSION,
    ChannelGroup,
    DatasetIndex,
    IndexError_,
    Store,
    UnsupportedFormatVersion,
    read_index,
)
from .transport import (
    PyfetchTransport,
    Response,
    Transport,
    TransportError,
    UrllibTransport,
    default_transport,
    running_in_pyodide,
)

__version__ = "0.1.0.dev0"

#: Names that live behind the ``zarr`` extra. Resolved on first use rather than imported
#: here, because importing them would make the base install require zarr, and the base
#: install exists precisely to require nothing.
_ZARR_EXTRA = {
    "NemarHttpStore": "eegprep_lean.store",
    "ReadOnlyStoreError": "eegprep_lean.store",
    "Window": "eegprep_lean.window",
    "open_array": "eegprep_lean.store",
    "read_window": "eegprep_lean.window",
    "to_physical": "eegprep_lean.window",
}


def __getattr__(name: str):
    """Load a zarr-backed name on demand, and say which extra supplies it if it is absent."""
    module_name = _ZARR_EXTRA.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise ImportError(
            f"{name} needs the zarr extra: pip install 'eegprep-lean[zarr]'. In a browser, "
            'micropip.install("zarr==3.4.0", deps=False), because zarr pins numcodecs and '
            "numcodecs publishes no emscripten wheel at any version."
        ) from err
    return getattr(module, name)


__all__ = [
    "SUPPORTED_FORMAT_VERSION",
    "NemarHttpStore",
    "ReadOnlyStoreError",
    "Window",
    "ChannelGroup",
    "DatasetIndex",
    "IndexError_",
    "PyfetchTransport",
    "Response",
    "Store",
    "Transport",
    "TransportError",
    "UnsupportedFormatVersion",
    "UrllibTransport",
    "__version__",
    "default_transport",
    "open_array",
    "read_index",
    "read_window",
    "running_in_pyodide",
    "to_physical",
]
