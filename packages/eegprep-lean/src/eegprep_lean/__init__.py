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

#: Names supplied by an extra, and which extra supplies each. Resolved on first use
#: rather than imported here, because importing them would make the base install require
#: their dependencies, and the base install exists precisely to require none.
_EXTRA_NAMES = {
    "NemarHttpStore": ("eegprep_lean.store", "zarr"),
    "ReadOnlyStoreError": ("eegprep_lean.store", "zarr"),
    "open_array": ("eegprep_lean.store", "zarr"),
    "read_window": ("eegprep_lean.window", "zarr"),
    # numpy, not zarr: window.py imports zarr only inside read_window, so these two work
    # under the plot extra alone.
    "Window": ("eegprep_lean.window", "plot"),
    "to_physical": ("eegprep_lean.window", "plot"),
    "default_spacing": ("eegprep_lean.plot", "plot"),
    "plot_window": ("eegprep_lean.plot", "plot"),
    "to_png": ("eegprep_lean.plot", "plot"),
}

#: What to tell someone who has the name but not the dependency. The browser line is the
#: part worth keeping: zarr's `numcodecs>=0.14` pin is metadata, and numcodecs publishes
#: no emscripten wheel at any version, so a normal resolve fails on a dependency these
#: stores never use at runtime.
_EXTRA_HINTS = {
    "zarr": (
        "pip install 'eegprep-lean[zarr]'. In a browser, "
        'micropip.install("zarr==3.4.0", deps=False), because zarr pins numcodecs and '
        "numcodecs publishes no emscripten wheel at any version."
    ),
    "plot": (
        "pip install 'eegprep-lean[plot]'. In a browser, micropip.install(\"matplotlib\"), which Pyodide bundles."
    ),
}


#: The top-level packages each extra actually installs. Used to tell "the extra is not
#: installed" apart from "the module that needed it is broken", which look identical from
#: outside and have opposite answers.
_EXTRA_ROOTS = {
    "zarr": frozenset({"zarr", "numpy"}),
    "plot": frozenset({"matplotlib", "numpy"}),
}


def _is_missing_extra(err: ImportError, extra: str) -> bool:
    """True when this ImportError is the extra being absent, rather than a bug in here.

    ``ImportError.name`` is the module that could not be imported, which the interpreter
    sets for both a missing module and a missing name within one.
    """
    return (err.name or "").split(".")[0] in _EXTRA_ROOTS[extra]


def __getattr__(name: str):
    """Load a name from its extra on demand, and say which extra supplies it if absent."""
    entry = _EXTRA_NAMES.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, extra = entry
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        # Only when the extra itself is what is missing. An ImportError from inside our
        # own module is a bug here, and reporting it as a missing extra sends the reader
        # off to install something they already have while the real error is buried.
        if not _is_missing_extra(err, extra):
            raise
        raise ImportError(f"{name} needs the {extra} extra: {_EXTRA_HINTS[extra]}") from err
    return getattr(module, name)


__all__ = [
    "SUPPORTED_FORMAT_VERSION",
    "ChannelGroup",
    "DatasetIndex",
    "IndexError_",
    "NemarHttpStore",
    "PyfetchTransport",
    "ReadOnlyStoreError",
    "Response",
    "Store",
    "Transport",
    "TransportError",
    "UnsupportedFormatVersion",
    "UrllibTransport",
    "Window",
    "__version__",
    "default_spacing",
    "default_transport",
    "open_array",
    "plot_window",
    "read_index",
    "read_window",
    "running_in_pyodide",
    "to_physical",
    "to_png",
]
