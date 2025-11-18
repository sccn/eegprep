Project Cleanup
===============

- document role of files in project root or remove: 
   - `config.json.example`
   - `install.sh`
   - `main`
   - `main.py`
   - `out_dir/`

- remove or consolidate into a common location the various 
  developers' personal test scripts:
  - `notebooks/` (some)
  - `scripts/` (some)

- make sure eegprep can be pip installed without necessarily pulling 
  in >7GB of CUDA binaries on Linux (requires cpu-only build of torch, 
  which may require a different install framework than what's used here right now)
