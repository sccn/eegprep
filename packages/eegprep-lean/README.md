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
The index reader works against the live archive.
Reading array data is not implemented yet.

## Install

| install | download | what it buys |
|---|---|---|
| `eegprep-lean` | 4.2 MB | read a window of data |
| `eegprep-lean[plot]` | 14.0 MB | and draw it |
| `eegprep-lean[preprocess]` | 30.4 MB | and filter and resample it |

Measured against the Pyodide 0.29.5 distribution.
Plotting is deliberately not in the base:
it is the first thing a person asks for after looking at data, and it costs 9.8 MB,
so it arrives when a plot is actually asked for.

In a browser, zarr needs `micropip.install("zarr==3.4.0", deps=False)`.
Its `numcodecs>=0.14` pin is metadata, and numcodecs publishes no emscripten wheel at any
version, so a normal resolve fails on a dependency these stores never use at runtime.

## Use

```python
from eegprep_lean import read_index

index = await read_index("nm000103")
store = index.store("sub-NDARAA075AMK/eeg/sub-NDARAA075AMK_task-DespicableMe_eeg.set")
group = store.group()          # 129 channels at 250 Hz
url = index.level0_url(store)  # full resolution, built from the contract's own layout
```

## Two things that will surprise you

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

## Tests

```bash
pytest -m "not network"   # offline, against a captured real index document
pytest -m network         # reaches zarr.nemar.org
```

The network tier is not optional coverage.
What is under test is conformance to a contract another team serves, so a stand-in that
always agrees with this reader would prove nothing about it.

[adr0069]: https://github.com/nemarOrg/nemar-cli/blob/main/.context/decisions/0069-the-browser-runtime-is-its-own-package-and-eegprep-stays-whole.md
[contract]: https://github.com/nemarOrg/nemar-cli/blob/main/.context/eegprep_lean_contract.md
