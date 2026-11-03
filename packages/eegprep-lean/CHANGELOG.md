# Changelog

## 0.4.1

First version number shared with eegprep. The previous package version was 0.1.0.dev3.

- `eegprep_lean.help()` prints each public function with a one-line summary from its docstring, the extra that supplies it, what is not available yet, and a short example. The text is read from source, so calling it does not import an extra.
- `dir(eegprep_lean)` lists the lazy public names, including names that live behind an extra, without importing them.
