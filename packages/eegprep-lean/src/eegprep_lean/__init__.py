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

__all__ = [
    "SUPPORTED_FORMAT_VERSION",
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
    "read_index",
    "running_in_pyodide",
]
