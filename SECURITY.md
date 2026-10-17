# Security policy

## Reporting a vulnerability

Please do not open a public issue, pull request or discussion for a security problem.

Email **suranganath@ucsd.edu** with:

- the EEGPrep version or commit;
- what an attacker controls (for example a crafted `.set`/`.fdt` file, a BIDS dataset, or a pipeline config) and what they gain;
- a minimal reproducer, ideally a script that generates the malicious input;
- a proposed fix, if you have one.

Never send real participant EEG data.

## Supported versions

EEGPrep is alpha software. Fixes go into the `develop` branch and ship in the next release on PyPI. Only the latest release and `develop` are supported; older releases do not get backported fixes.

| Version | Supported |
| --- | --- |
| `develop` | yes |
| latest PyPI release (currently 0.4.x) | yes |
| older releases | no |

## What to expect

- We acknowledge your report within 5 working days.
- We confirm or rule out the issue and tell you what we plan to do, usually within 30 days.
- We fix it on `develop`, add a regression test, and release. We credit you in the changelog unless you ask us not to.
- Please keep the details private until a fixed release is out, or until we agree on a date with you.

## Scope

In scope are EEGPrep's own Python code and the way it uses its dependencies. The vendored MATLAB EEGLAB code under `src/eegprep/eeglab` belongs to [EEGLAB](https://github.com/sccn/eeglab); report problems there to the EEGLAB maintainers. Numerical differences from MATLAB EEGLAB are bugs, not security issues; please file them as ordinary issues.

`.oss-scanner/threat_model.md` describes in more detail which inputs we treat as untrusted and how we rate severity.
