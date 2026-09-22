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
| `eegprep-lean` | 1 | 26 KB | read the index and the channel list: what a dataset holds, at what rate, in what units |
| `eegprep-lean[zarr]` | 12 | 4.2 MB | and read a window of signal |
| `eegprep-lean[zarr,plot]` | 22 | 14.0 MB | and draw it |
| `eegprep-lean[zarr,plot,preprocess]` | 23 | 30.4 MB | and filter and resample it |

ADR 0069's measurements against the Pyodide 0.29.5 distribution,
counting packages as micropip installs them there, transitive dependencies included.
The base row is this package's own wheel, which is pure Python and declares no
dependencies, so it is the whole cost of that tier.
The rows are cumulative, each adding to the one above it,
which is why the last row names `plot` as well:
30.4 MB was measured with matplotlib present,
and `[zarr,preprocess]` on its own has not been measured.
Plotting is deliberately not in the base:
it is the first thing a person asks for after looking at data, and it costs 9.8 MB,
so it arrives when a plot is actually asked for.

**zarr is an extra rather than a base dependency, which differs from ADR 0069's table.**
Not a trim for its own sake.
zarr installs under Pyodide only as `micropip.install("zarr==3.4.0", deps=False)`:
its `numcodecs>=0.14` pin is metadata,
and numcodecs publishes no emscripten wheel at any version,
so a normal resolve fails on a dependency these stores never use at runtime.
A base that declared zarr would therefore make `micropip.install("eegprep-lean")` fail
outright in the browser this package exists for.
The tier the ADR measured at 4.2 MB is `[zarr]`;
the base below it is this package alone, and it reads the index without installing
anything.

## Use

With nothing installed but this package, you can ask what a dataset holds:

```python
from eegprep_lean import read_group_metadata, read_index

index = await read_index("nm000103")
store = index.stores[0]

group = await read_group_metadata(index, store)
group.rate, group.original_rate     # 250.0, 500.0: level 0 is resampled
group.channels[0].label             # "E1"
group.channels[0].unit              # "uV"
```

Reading a window of signal needs the `zarr` extra:

```python
from eegprep_lean import read_index, read_window

index = await read_index("nm000103")
store = index.store("sub-NDARAA075AMK/eeg/sub-NDARAA075AMK_task-DespicableMe_eeg.set")

window = await read_window(index, store, start_sample=2500, n_samples=500, channels=[0, 1, 2])
window.data.shape  # (3, 500), physical units
window.times_s[0]  # 10.0, seconds into the recording, not into the window
```

The window carries the channel labels and the unit the group declares,
so `window.unit` is `"uV"` here and `window.labels` is `("E1", "E2", "E3")`,
read from the store rather than assumed.
It reads only the chunks the window spans:
the store is sharded, so a one-second window of one channel costs a shard index read and
one inner chunk, not the whole 7 MB shard.

To draw it, with the `plot` extra:

```python
from eegprep_lean import plot_window, to_png

ax = plot_window(window)  # stacked traces, first channel at the top
png = to_png(ax.figure)  # bytes, which is what a browser can show
```

Traces are demeaned for display, because level-0 channels carry per-channel offsets in
the thousands that differ between channels by more than the signal spans;
without it every trace is a flat line at its own level.
That never touches the window, and `demean=False` draws the values as they are.

## In a runtime that owns the network

In a plain Pyodide, reads go through `pyodide.http.pyfetch`.
A sandboxed runtime may remove that module and offer its own client instead,
one that enforces what executed code is allowed to reach.
The runtime registers a transport over that client once, before any reader code runs,
and everything above works unchanged:

```python
import eegprep_lean

# `client` is the host's: awaited as client(url, headers=...), returning (status, body)
eegprep_lean.set_default_transport(eegprep_lean.FetchTransport(client))
```

`FetchTransport` sends `Range` and no other header,
and accepts only a `2xx`:
any other status is a `TransportError`, a redirect included, since it follows none,
and so is a range request answered in full.
A request the host refuses must raise rather than answer with a status of its own,
because the store reads 403, 404 and 416 as a key that does not exist.
An explicit `transport=` argument still wins over the registered default,
and `set_default_transport(None)` restores platform selection.

## Five things that will surprise you

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

**The unit is on the channel group, not on the array.**
The level-0 array declares its conversion, `physical = digital * scale + offset`, per
channel, but not the unit that conversion produces,
so reading only the array leads to the conclusion that units are unknowable.
They are not.
The channel group lists every channel with its own `label` and `unit`, and `read_window`
reads them, so a window says what it is measured in and what its channels are called.
It reports a unit only when every channel in the window agrees on one,
because the store contract says to read the unit from the channel rather than from the
modality,
and magnetoencephalography is a Tesla-based unit rather than a voltage.

**Level 0 is not always the rate the recording was acquired at.**
The store resamples it to `min(native rate, modality cap)`,
and the cap for electroencephalography is 250 Hz,
so `nm000103` was recorded at 500 Hz and arrives here at 250.
`Window.original_rate` says so, and a plot puts it in the title.
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
