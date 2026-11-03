# 0071. A plain install is the beginner pipeline

- Status: Accepted
- Date: 2026-11-03
- Deciders: Suraj, after Arno's reply the same day
- Supersedes: the "default install stays whole" clause of [ADR 0069](https://github.com/nemarOrg/nemar-cli/blob/main/.context/decisions/0069-the-browser-runtime-is-its-own-package-and-eegprep-stays-whole.md) (accepted 2026-09-20, owner Seyed Yahya Shirazi, in nemar-cli), only for the libraries the default pipeline does not import. The rest of ADR 0069 stays in force: `eegprep-lean` remains its own distribution. ADR 0070, the browser recipe, is unchanged.
- Number: 0071, so this file does not collide with ADR 0070. It lives in this repository because `.context/` is gitignored here.

## Decision

A plain `pip install eegprep` installs a pipeline a beginner can run. No brackets.

The base requires numpy, scipy, matplotlib, psutil, and pyedflib. It also keeps h5py, because `pop_loadset` imports it while sniffing an EEGLAB `.set` file, and the small packages already required for that path: colorama, eeglabio, packaging, pyyaml, and threadpoolctl. Matplotlib was already a direct dependency. It does not arrive through mne, and plots have to keep working after mne leaves the base.

Loading, cleaning (`clean_artifacts`), plotting (`topoplot`), a RAM reservation such as `4GB`, and EDF, BDF, and GDF export (`pop_writeeeg`) work from that install.

Heavy pieces the default preprocessing path and the common plots do not import are named extras:

| Extra | Installs | Used for |
| --- | --- | --- |
| `eeglab` | oct2py | The Octave engine: `pop_resample(engine="octave")`, `iclabel(engine="octave")`, the RANSAC octave subroutine, `eeglab_clean_artifacts` |
| `mne` | mne | Conversion to and from MNE, and the EDF, BDF, and GDF readers. EEGLAB `.set`, CNT, and BrainVision do not need it |
| `pybids` | pybids | BIDS import and `bids_preproc` |
| `picard` | python-picard | Picard ICA. The default ICA is `runica`, which does not import it. This extra also installs scikit-learn; eegprep does not import scikit-learn itself |

`pip install "eegprep[full]"` is those four extras. On top of the base, that is today's complete install: the required set on `develop` at the time of this decision (numpy, scipy, matplotlib, h5py, mne, pybids, python-picard, oct2py, psutil, pyedflib, plus colorama, eeglabio, packaging, pyyaml, and threadpoolctl).

`eegprep[gui]`, `eegprep[console]`, `eegprep[torch]`, and `eegprep[docs]` stay as they are. `eegprep[all]` includes them and the four extras above.

A missing extra raises `ImportError`. The message names the library, the feature, `pip install "eegprep[<extra>]"`, and `pip install "eegprep[full]"`. `eegprep.help()` prints the extra next to each public name that needs one, read from source, without importing the module.

There is no major version bump. Moving libraries the default path does not import does not change what a beginner gets from `pip install eegprep`: a working pipeline.

## eegprep-lean

`eegprep-lean` stays a separate package for NEMAR and OSA. It is not an extra of `eegprep`, and `pip install eegprep` does not install it. NEMAR pins `eegprep_lean-0.1.0.dev3-py3-none-any.whl` (28,537 bytes) by sha256. That pin keeps working. There is no alias that makes the lean name install `eegprep`.

## Synced releases

eegprep and eegprep-lean are released together, with one shared version number. The release script builds both, publishes both, and fails if their versions differ. A release that moves one package and not the other is a failed release.

## Why

Arno, 2026-11-03: the user base is not savvy enough for the bracket calls. Usability is the most important thing. People and beginners do not care about the peculiarities of the installer. There should still be a more streamlined release, and a release script that releases both packages at the same time with synced version numbers.

Suraj accepted that. A beginner runs `pip install eegprep` and gets the pipeline. Brackets are for Octave, MNE, BIDS, and Picard, which that pipeline does not need.

## Rejected alternatives

Zero-dependency base, with `eegprep[core]` for numpy and scipy. A plain install would not clean or plot. Beginners will not add the brackets, so the install they actually run would not be usable. That shape is also what would force a major version, because it changes `pip install eegprep` from a pipeline into an empty package. Rejected.

An alias that makes `eegprep-lean` install `eegprep` once the base is empty. NEMAR and OSA need the separate small wheel. The accepted base is a scientific install, not that 28 KB wheel. Publishing the alias would replace the pin. Rejected. The separate package stays.

A major version bump to carry an empty base. The accepted change does not empty the base, so it does not need a new major version. `eegprep[full]` is how an existing complete install is spelled. Rejected as unnecessary.
