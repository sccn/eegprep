# Pyodide browser execution and Phase 2 benchmark

This page records the Phase 2 gate for the [browser epic](https://github.com/sccn/eegprep/issues/324).
The harness installs a wheel built from the working tree under Pyodide, runs a
real continuous-data smoke pipeline, and measures the ICA and matrix products
that determine whether later browser work is worthwhile.

## Runtime boundary

The harness pins [Pyodide 0.29.5](https://pyodide.org/en/stable/usage/faq.html)
and uses Pyodide's `emfs:` local-wheel transport. The package-resolution path
pre-installs a universal `docopt==0.6.2` wheel built from a hash-checked sdist
because the PyPI release is an sdist. The live harness also pins the
Pyodide-compatible pure Python versions `mne==1.10.0`, `sympy==1.14.0`, and
`threadpoolctl==3.6.0` while
leaving EEGPrep's published dependency ranges unchanged.
The smoke install completed successfully with this closure; no additional
sdist-only blocker surfaced in the live harness.

Pyodide does not provide pthread support, so one Pyodide instance cannot run
four or eight Python threads for a single ICA call. The benchmark therefore
limits native BLAS to one thread as well. A browser [Web Worker](https://pyodide.org/en/stable/usage/index.html)
is still the correct execution boundary: it keeps the UI responsive, but it
does not make one ICA solve faster.

The scalable design is a bounded worker pool for independent jobs. EEGPrep
should own the worker-safe operation contract and serialized progress/result
messages. The NEMAR/OSA host should own worker lifecycle, queueing, resource
limits, cancellation policy, and whether a request uses one worker or a pool.
Splitting one existing `runica` or Picard solve across workers is out of scope:
its iterative weights are global state and would require a new synchronized
distributed algorithm and a separate parity gate.

## Reproducing the gate

The harness requires `uv`, Node.js 22 or newer, and npm. From the repository
root, build the EEGPrep wheel and run the sample-data smoke test with:

```bash
uv build --wheel --out-dir pyodide-artifacts
tools/pyodide/run_pyodide.sh \
  --wheel pyodide-artifacts/eegprep-*.whl \
  --script tools/pyodide/smoke.py \
  --sample-data-dir sample_data
```

The CI job runs the same smoke test, then benchmarks a fixed `(64, 15000)`
continuous array (64 channels, 60 seconds at 250 Hz). Each algorithm has one
warm-up and three measured runs, with a maximum of 512 ICA iterations and a
fixed seed. The matrix benchmark uses the two runica products:

* `(64, 64) @ (64, 49)` for block activations;
* `(64, 64) @ (64, 64)` for the square weight update.

Picard is measured through its underlying `picard(..., return_n_iter=True)` API
with the same algorithmic options as EEGPrep's `eeg_picard` wrapper; benchmark
output is quiet and includes the underlying iteration telemetry. The benchmark
does not change either production ICA implementation.

## Measured results

The following tables are generated from the native and Pyodide JSON reports
with `tools/pyodide/compare_benchmarks.py`. The raw reports and loader logs are
also uploaded by CI as the `pyodide-phase-2-output` artifact.

### ICA

| Algorithm | Native median (s) | Native median iterations | Native converged | Pyodide median (s) | Pyodide median iterations | Pyodide converged |
| --- | ---: | ---: | :---: | ---: | ---: | :---: |
| runica | 86.900706 | 512.0 | false | 351.139361 | 512.0 | false |
| picard | 7.699578 | 511.0 | false | 65.949396 | 511.0 | false |

### Matrix multiplication

| runica product | dtype | Native NumPy `@` (s) | Native BLAS (s) | Native speedup | Pyodide NumPy `@` (s) | Pyodide BLAS (s) | Pyodide speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| activation | float64 | 0.000016 | 0.000018 | 0.88x | 0.000239 | 0.000106 | 2.26x |
| activation | float32 | 0.000009 | 0.000015 | 0.62x | 0.000251 | 0.000246 | 1.02x |
| weight_update | float64 | 0.000011 | 0.000018 | 0.61x | 0.000308 | 0.000127 | 2.43x |
| weight_update | float32 | 0.000008 | 0.000015 | 0.53x | 0.000311 | 0.000150 | 2.07x |

### Decisions

The comparison script records whether Picard remains a safe browser default
and whether SciPy BLAS is at least 1.2x faster than NumPy `@` for every tested
runica product and dtype. Those decisions are intentionally based on measured
convergence and timings, not on the code path alone.

For this run, Picard was much faster in wall-clock time but emitted its
non-convergence warning at 511 iterations; runica also reached the 512-step
cap. The conservative browser-default gate therefore returns **false** for
Picard. The BLAS gate also returns **false**: the float32 activation product
was only 1.02x faster with `sgemm`, below the 1.2x threshold. Phase 3 should
remain a documentation/follow-up decision rather than a production helper
based on this measurement.
