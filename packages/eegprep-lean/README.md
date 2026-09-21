# eegprep-lean

EEGPrep's browser runtime.
Reads a NEMAR recording without installing EEGPrep.

`pip install eegprep` is unchanged and installs everything it always did.
This is a second distribution, built to a download budget instead,
because a browser session pays for its dependencies before it sees anything.
The reasoning is [Architecture Decision Record 0069][adr0069],
and what this package guarantees, and where it deliberately differs from EEGPrep, is its
[contract][contract].

**Status: under construction.**
Reading an index, reading a window of signal, and drawing it all work against the live
archive.
Preprocessing is not implemented yet.

## Install

| install | packages | download | what it buys |
|---|---|---|---|
| `eegprep-lean` | 1 | this package | read the index: what a dataset holds, at what rate, and where |
| `eegprep-lean[zarr]` | 12 | 4.2 MB | and read a window of signal |
| `eegprep-lean[zarr,plot]` | 22 | 14.0 MB | and draw it |
| `eegprep-lean[zarr,preprocess]` | 23 | 30.4 MB | and filter and resample it |

Measured against the Pyodide 0.29.5 distribution.
Plotting is deliberately not in the base:
it is the first thing a person asks for after looking at data, and it costs 9.8 MB,
so it arrives when a plot is actually asked for.

**zarr is an extra rather than a base dependency, which differs from ADR 0069's table.**
Not a trim for its own sake.
zarr installs under Pyodide only as `micropip.install("zarr==3.4.0", deps=False)`:
its `numcodecs>=0.14` pin is metadata, and numcodecs publishes no emscripten wheel at any
version, so a normal resolve fails on a dependency these stores never use at runtime.
A base that declared zarr would therefore make `micropip.install("eegprep-lean")` fail
outright in the browser this package exists for.
The tier the ADR measured at 4.2 MB is `[zarr]`;
the base below it is this package alone, and it reads the index without installing
anything.

## Use

```python
from eegprep_lean import read_index, read_window

index = await read_index("nm000103")
store = index.store("sub-NDARAA075AMK/eeg/sub-NDARAA075AMK_task-DespicableMe_eeg.set")

window = await read_window(
    index, store, start_sample=2500, n_samples=500, channels=[0, 1, 2]
)
window.data.shape   # (3, 500), physical units
window.times_s[0]   # 10.0, seconds into the recording, not into the window
```

`read_window` needs the `zarr` extra.
It reads only the chunks the window spans:
the store is sharded, so a one-second window of one channel costs a shard index read and
one inner chunk, not the whole 7 MB shard.

To draw it, with the `plot` extra:

```python
from eegprep_lean import plot_window, to_png

ax = plot_window(window)          # stacked traces, first channel at the top
png = to_png(ax.figure)           # bytes, which is what a browser can show
```

Traces are demeaned for display, because level-0 channels carry per-channel offsets in
the thousands that differ between channels by more than the signal spans;
without it every trace is a flat line at its own level.
That never touches the window, and `demean=False` draws the values as they are.

## Four things that will surprise you

**Everything that touches the network is `async`, and there is no synchronous wrapper.**
Not a style choice.
Zarr's synchronous API starts an IO thread, Pyodide's main thread cannot start one, and
the resulting `RuntimeError: can't start new thread` names neither zarr nor the browser.
Offering a synchronous wrapper would mean starting the loop that cannot be started.

**Reads go through `zarr.nemar.org`, and never through the bucket.**
The index also publishes `data_base`, naming where the bytes physically sit today.
This package does not follow it.
`zarr.nemar.org` is the contract; the redirect behind it is an implementation detail that
is allowed to move, and a client holding the bucket URL is holding something that was
never promised to keep working, while making its reads invisible to the archive serving
them.

**Plotting never touches `matplotlib.pyplot`.**
pyplot selects a backend when it is imported, and in a browser that backend wants the
DOM.
The object-oriented `Figure` API needs none, and it is the shape the browser case wants
anyway: a PNG to hand to the page rather than a window to show.
A test asserts pyplot stays unimported, in a subprocess, because that failure is
invisible natively and fatal in Pyodide.

**The physical units are not knowable from here.**
The array declares its conversion, `physical = digital * scale + offset`, per channel,
but not the units that conversion produces;
the index reports only that a `channels.tsv` sidecar supplied them.
So a window holds values in the recording's own units and does not claim to know which.
Assuming microvolts is the kind of guess that is right often enough to go unquestioned
and wrong quietly when it is not.
Large offsets are normal: level 0 is raw signal before referencing or filtering.

## Tests

```bash
pytest -m "not network"   # offline, against a captured real index document
pytest -m network         # reaches zarr.nemar.org
```

Each extra is also tested on its own in CI, not only together,
because `window.py` imports zarr inside `read_window` rather than at module scope so the
`plot` tier stands alone;
a stray module-scope import would otherwise surface only in a browser.
The network tier runs on a schedule in `lean-live.yml` rather than on pull requests,
so an outage at the archive cannot turn an unrelated pull request red.

The network tier is not optional coverage.
What is under test is conformance to a contract another team serves, so a stand-in that
always agrees with this reader would prove nothing about it.

[adr0069]: https://github.com/nemarOrg/nemar-cli/blob/main/.context/decisions/0069-the-browser-runtime-is-its-own-package-and-eegprep-stays-whole.md
[contract]: https://github.com/nemarOrg/nemar-cli/blob/main/.context/eegprep_lean_contract.md
