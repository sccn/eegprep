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

``eegprep_lean.help()`` prints every public name, which extra supplies it, and what is not
available yet.
"""

import ast
import functools
import pathlib
import re
import textwrap

from .channels import Channel, GroupMetadata, read_group_metadata
from .extras import EXTRA_HINTS, EXTRA_ROOTS, is_missing_extra, missing_extra_error
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
    FetchTransport,
    PyfetchTransport,
    Response,
    Transport,
    TransportError,
    UrllibTransport,
    default_transport,
    running_in_pyodide,
    set_default_transport,
)

__version__ = "0.1.0.dev3"

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
        if not is_missing_extra(err, extra):
            raise
        raise missing_extra_error(name, extra, err) from err
    return getattr(module, name)


def __dir__() -> list[str]:
    """Include the names behind an extra, which tab completion would otherwise never offer."""
    return sorted({*globals(), *_EXTRA_NAMES})


_ICLABEL_ISSUE = "https://github.com/sccn/eegprep/issues/378"

_RST_MARKUP = re.compile(r"(?::\w+:)?`+~?")

_EXAMPLE = """\
from eegprep_lean import plot_window, read_index, read_window, to_png

index = await read_index("nm000103")
window = await read_window(index, index.stores[0], start_sample=2500, n_samples=500, channels=[0, 1, 2])
png = to_png(plot_window(window).figure)"""


@functools.cache
def _module_source(module: str) -> ast.Module:
    return ast.parse(pathlib.Path(__file__).with_name(module.rpartition(".")[2] + ".py").read_text(encoding="utf-8"))


def _definition(module: str, name: str) -> ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef:
    """Read from source rather than imported, so describing a name never needs its extra."""
    return next(
        node
        for node in _module_source(module).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def help() -> None:
    """Print each public name, what it does, the extra that supplies it, and an example."""
    functions: dict[str | None, list[tuple[str, str]]] = {None: [], **{extra: [] for extra in EXTRA_ROOTS}}
    classes: dict[str | None, list[str]] = {extra: [] for extra in functions}
    for name in __all__:
        if name in _EXTRA_NAMES:
            module, extra = _EXTRA_NAMES[name]
        elif callable(globals()[name]):
            module, extra = globals()[name].__module__, None
        else:
            continue
        node = _definition(module, name)
        if isinstance(node, ast.ClassDef):
            classes[extra].append(name)
            continue
        call = f"await {name}()" if isinstance(node, ast.AsyncFunctionDef) else f"{name}()"
        summary = _RST_MARKUP.sub("", (ast.get_docstring(node) or "").partition("\n")[0])
        functions[extra].append((call, summary))

    width = max(len(call) for entries in functions.values() for call, _ in entries)
    lines = [f"eegprep-lean {__version__}: read a NEMAR recording, natively or in a browser.", ""]
    for extra, entries in functions.items():
        heading = (
            "Base install, no extra: pip install eegprep-lean"
            if extra is None
            else f"{extra} extra: {EXTRA_HINTS[extra]}"
        )
        lines.append(textwrap.fill(heading, width=88, subsequent_indent="    "))
        lines += [f"  {call:<{width}}  {summary}" for call, summary in entries]
        if classes[extra]:
            lines.append(
                textwrap.fill(
                    "Classes: " + ", ".join(classes[extra]), 88, initial_indent="  ", subsequent_indent="    "
                )
            )
        lines.append("")
    lines += [
        "preprocess extra: installs scipy, and supplies nothing yet.",
        "",
        "Not available yet:",
        "  Preprocessing: filtering and resampling are not implemented.",
        f"  ICLabel component classification: tracked in {_ICLABEL_ISSUE}",
        "",
        "Example, with the zarr and plot extras:",
        textwrap.indent(_EXAMPLE, "  "),
        "",
        "help(eegprep_lean.<name>) shows the full docstring; a name behind an extra needs that extra installed.",
    ]
    print("\n".join(lines))


# Not in __all__: a star import would replace the builtin help() with this one.
__all__ = [
    "EXTRA_HINTS",
    "EXTRA_ROOTS",
    "SUPPORTED_FORMAT_VERSION",
    "Channel",
    "ChannelGroup",
    "DatasetIndex",
    "FetchTransport",
    "GroupMetadata",
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
    "is_missing_extra",
    "missing_extra_error",
    "open_array",
    "plot_window",
    "read_group_metadata",
    "read_index",
    "read_window",
    "running_in_pyodide",
    "set_default_transport",
    "to_physical",
    "to_png",
]
