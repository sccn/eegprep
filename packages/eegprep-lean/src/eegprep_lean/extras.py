"""What to tell someone who has a name but not the dependency behind it.

One place, because the same message has to come from two. The lazy loader in
``__init__`` catches an extra missing at *attribute access*, which is where most names
are reached. :func:`~eegprep_lean.window.read_window` is different: it imports zarr
inside its own body, so the module imports cleanly under the ``plot`` extra alone and the
failure happens at call time, long past the loader. Without this, that one name, the one
most likely to be called in exactly that half-installed state, reports a bare
``ModuleNotFoundError`` from inside another module while every other name explains itself.
"""

from __future__ import annotations

#: The top-level packages each extra actually installs. Used to tell "the extra is not
#: installed" apart from "the module that needed it is broken", which look identical from
#: outside and have opposite answers.
EXTRA_ROOTS = {
    "zarr": frozenset({"zarr", "numpy"}),
    "plot": frozenset({"matplotlib", "numpy"}),
}

#: The browser line is the part worth keeping: zarr's ``numcodecs>=0.14`` pin is metadata,
#: and numcodecs publishes no emscripten wheel at any version, so a normal resolve fails
#: on a dependency these stores never use at runtime.
EXTRA_HINTS = {
    "zarr": (
        "pip install 'eegprep-lean[zarr]'. In a browser, "
        'micropip.install("zarr==3.4.0", deps=False), because zarr pins numcodecs and '
        "numcodecs publishes no emscripten wheel at any version."
    ),
    "plot": (
        "pip install 'eegprep-lean[plot]'. In a browser, micropip.install(\"matplotlib\"), which Pyodide bundles."
    ),
}


def is_missing_extra(err: ImportError, extra: str) -> bool:
    """True when this ImportError is the extra being absent, rather than a bug in here.

    ``ImportError.name`` is the module that could not be imported, which the interpreter
    sets for both a missing module and a missing name within one.
    """
    return (err.name or "").split(".")[0] in EXTRA_ROOTS[extra]


def missing_extra_error(name: str, extra: str, err: ImportError) -> ImportError:
    """The message every name behind an extra gives when its dependency is absent."""
    return ImportError(f"{name} needs the {extra} extra: {EXTRA_HINTS[extra]}")
