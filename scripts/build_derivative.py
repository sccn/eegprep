#!/usr/bin/env python3
"""Build a BIDS derivative dataset with Docker.

Automates: build Docker image, run preprocessing, copy reproducibility
files into the derivative's code/ folder, validate BIDS compliance.

Usage:
    python build_derivative.py /path/to/bids_dataset
    python build_derivative.py /path/to/bids_dataset --srate 200 --highpass 1.0
    python build_derivative.py /path/to/bids_dataset --skip-docker-build
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

EEGPREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_NAME = 'eegprep-minimal'


def run(cmd, **kwargs):
    """Run a shell command, print it, and check for errors."""
    print(f'  $ {cmd}')
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f'  ERROR: command exited with code {result.returncode}')
        sys.exit(result.returncode)
    return result


def docker_is_running():
    result = subprocess.run('docker info', shell=True,
                            capture_output=True, timeout=10)
    return result.returncode == 0


def build_docker():
    """Build the Docker image from the eegprep repo."""
    print('\n=== Building Docker image ===')
    run(f'docker build -t {IMAGE_NAME} {EEGPREP_ROOT}')


def run_docker(bids_root, srate, highpass, extra_args=''):
    """Run the Docker container on the dataset."""
    print('\n=== Running preprocessing ===')
    cmd = (f'docker run --rm -v {bids_root}:/data {IMAGE_NAME}'
           f' --srate {srate} --highpass {highpass}')
    if extra_args:
        cmd += f' {extra_args}'
    run(cmd)


def export_docker_image(dest):
    """Save the Docker image as a compressed tar."""
    print('\n=== Exporting Docker image ===')
    run(f'docker save {IMAGE_NAME} | gzip > {dest}')
    size_mb = os.path.getsize(dest) / 1048576
    print(f'  Image saved: {dest} ({size_mb:.0f} MB)')


def create_source_archive(dest):
    """Create a source archive excluding eeglab, bin, pycache, temp files."""
    print('\n=== Creating source archive ===')
    src_dir = os.path.join(EEGPREP_ROOT, 'src', 'eegprep')
    excludes = {'eeglab', 'bin', '__pycache__'}
    skip_files = {'tmp.set', 'tmp2.fdt'}

    with tarfile.open(dest, 'w:gz') as tar:
        # add src/eegprep (filtered)
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = [d for d in dirs if d not in excludes]
            for f in files:
                if f in skip_files or f.endswith('.pyc'):
                    continue
                full = os.path.join(root, f)
                arcname = os.path.relpath(full, EEGPREP_ROOT)
                tar.add(full, arcname=arcname)
        # add build files
        for name in ('pyproject.toml', 'LICENSE', 'README.md'):
            path = os.path.join(EEGPREP_ROOT, name)
            if os.path.exists(path):
                tar.add(path, arcname=name)

    size_mb = os.path.getsize(dest) / 1048576
    print(f'  Source archive: {dest} ({size_mb:.1f} MB)')


def save_pinned_requirements(dest):
    """Extract pinned requirements from the Docker image."""
    print('\n=== Saving pinned requirements ===')
    result = subprocess.run(
        f'docker run --rm --entrypoint pip {IMAGE_NAME} freeze',
        shell=True, capture_output=True, text=True)
    lines = [l for l in result.stdout.splitlines() if not l.startswith('eegprep @')]
    with open(dest, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  {len(lines)} packages pinned to {dest}')


def copy_code_folder(bids_root, deriv_dir, srate, highpass):
    """Populate the derivative's code/ folder with reproducibility files."""
    print('\n=== Populating code/ folder ===')
    code_dir = os.path.join(deriv_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)

    # Copy processing script
    src_script = os.path.join(EEGPREP_ROOT, 'scripts', 'bids_minimal_preproc.py')
    shutil.copy2(src_script, code_dir)

    # Copy Dockerfile (derivative version with adjusted paths)
    src_dockerfile = os.path.join(EEGPREP_ROOT, 'Dockerfile')
    dst_dockerfile = os.path.join(code_dir, 'Dockerfile')
    with open(src_dockerfile) as f:
        content = f.read()
    # Adjust paths: in code/ the script is at root level, not scripts/
    content = content.replace(
        'COPY scripts/bids_minimal_preproc.py',
        'COPY bids_minimal_preproc.py')
    with open(dst_dockerfile, 'w') as f:
        f.write(content)

    # Create source archive
    create_source_archive(os.path.join(code_dir, 'eegprep-source.tar.gz'))

    # Save pinned requirements
    save_pinned_requirements(os.path.join(code_dir, 'requirements.txt'))

    # Export Docker image
    export_docker_image(os.path.join(code_dir, 'eegprep-minimal.tar.gz'))

    # Read dataset_description for DOI
    ds_desc_path = os.path.join(bids_root, 'dataset_description.json')
    ds_name = os.path.basename(bids_root)
    ds_doi = ''
    ds_url = ''
    if os.path.exists(ds_desc_path):
        import json
        with open(ds_desc_path) as f:
            desc = json.load(f)
        ds_name = desc.get('Name', ds_name)
        ds_doi = desc.get('DatasetDOI', '')
        for src in desc.get('SourceDatasets', []):
            ds_url = src.get('URL', ds_url)

    # Write REPRODUCE.md
    reproduce = f"""# Reproducing the EEGPrep Derivative for {ds_name}

This derivative dataset was generated from **{ds_name}** using
[EEGPrep](https://github.com/sccn/eegprep) with minimal preprocessing:
resampling to {srate} Hz and highpass filtering at {highpass} Hz.

## 1. Download the source data
"""
    if ds_doi:
        reproduce += f"\n- **DOI:** {ds_doi}\n"
    if ds_url:
        reproduce += f"- **URL:** {ds_url}\n"

    reproduce += f"""
## 2. Install Docker

- **macOS:** https://docs.docker.com/desktop/install/mac-install/
- **Windows:** https://docs.docker.com/desktop/install/windows-install/
- **Linux:** https://docs.docker.com/engine/install/

## 3. Reproduce the derivative

### Option A: Use the pre-built Docker image (fastest)

```bash
docker load -i eegprep-minimal.tar.gz
docker run --rm -v /path/to/{os.path.basename(bids_root)}:/data eegprep-minimal
```

### Option B: Rebuild from source (any architecture)

```bash
tar xzf eegprep-source.tar.gz
docker build -t eegprep-minimal .
docker run --rm -v /path/to/{os.path.basename(bids_root)}:/data eegprep-minimal
```

## Processing details

| Parameter | Value |
|-----------|-------|
| Sampling rate | {srate} Hz |
| Highpass filter | {highpass} Hz (FIR Kaiser, transition band {highpass}--{highpass * 2} Hz) |
| Artifact rejection | Disabled |
| ICA / ICLabel | Disabled |
| Re-referencing | Disabled |

## Custom parameters

```bash
docker run --rm -v /path/to/dataset:/data eegprep-minimal --srate 200 --highpass 1.0
docker run --rm eegprep-minimal --help
```

## Reproduce without Docker

```bash
pip install eegprep
python bids_minimal_preproc.py --input /path/to/{os.path.basename(bids_root)}
```

## Files in this directory

| File | Purpose |
|------|---------|
| `REPRODUCE.md` | This file |
| `bids_minimal_preproc.py` | Generic processing script |
| `Dockerfile` | Docker build instructions |
| `requirements.txt` | Pinned Python dependencies |
| `eegprep-source.tar.gz` | EEGPrep source code |
| `eegprep-minimal.tar.gz` | Pre-built Docker image |
"""
    with open(os.path.join(code_dir, 'REPRODUCE.md'), 'w') as f:
        f.write(reproduce)

    print(f'  code/ folder ready: {code_dir}')


def prepend_readme_notice(deriv_dir):
    """Prepend a data availability notice to the derivative README."""
    readme_path = os.path.join(deriv_dir, 'README.md')
    # fall back to README (no extension) if .md doesn't exist
    if not os.path.exists(readme_path):
        readme_path = os.path.join(deriv_dir, 'README')
    notice = """## Data Availability and Regeneration Instructions

This is a derivative dataset. If any data are missing, you can use the
instructions in the code folder to download the raw data and regenerate
the derivatives.

Below is the README file of the raw data.

----------------------------------------

"""
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            existing = f.read()
        # don't prepend twice
        if 'Data Availability and Regeneration Instructions' not in existing:
            with open(readme_path, 'w') as f:
                f.write(notice + existing)
        else:
            with open(readme_path, 'w') as f:
                f.write(existing)
    else:
        with open(readme_path, 'w') as f:
            f.write(notice)
    print(f'  Updated {readme_path} with regeneration notice')


def add_bidsignore(deriv_dir):
    """Add .bidsignore for derivative-specific files."""
    path = os.path.join(deriv_dir, '.bidsignore')
    with open(path, 'w') as f:
        f.write('*_scans.tsv\n')
        f.write('*_coordsystem.json\n')
        f.write('code/\n')


def validate_bids(deriv_dir):
    """Run bids-validator if available."""
    print('\n=== Validating BIDS compliance ===')
    if shutil.which('bids-validator'):
        result = subprocess.run(
            f'bids-validator {deriv_dir}', shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        errors = result.stdout.count('[ERR]')
        warnings = result.stdout.count('[WARN]')
        print(f'  Result: {errors} errors, {warnings} warnings')
    else:
        print('  bids-validator not found, skipping (npm install -g bids-validator)')


def main():
    parser = argparse.ArgumentParser(
        description='Build a BIDS derivative dataset with Docker.')
    parser.add_argument('bids_root', help='Path to the BIDS dataset')
    parser.add_argument('--srate', type=float, default=100.0,
                        help='Target sampling rate (default: 100)')
    parser.add_argument('--highpass', type=float, default=0.5,
                        help='Highpass cutoff in Hz (default: 0.5)')
    parser.add_argument('--skip-docker-build', action='store_true',
                        help='Skip building the Docker image (reuse existing)')
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip running the pipeline (reuse existing output)')
    parser.add_argument('--extra-args', default='',
                        help='Extra args passed to the Docker container')
    args = parser.parse_args()

    bids_root = os.path.abspath(args.bids_root)
    if not os.path.isdir(bids_root):
        print(f'ERROR: {bids_root} is not a directory')
        sys.exit(1)

    if not docker_is_running():
        print('ERROR: Docker is not running. Start Docker Desktop first.')
        sys.exit(1)

    deriv_dir = os.path.join(bids_root, 'derivatives', 'eegprep')

    # Build
    if not args.skip_docker_build:
        build_docker()

    # Process
    if not args.skip_processing:
        run_docker(bids_root, args.srate, args.highpass, args.extra_args)

    # Add .bidsignore
    add_bidsignore(deriv_dir)

    # Prepend regeneration notice to derivative README
    prepend_readme_notice(deriv_dir)

    # Copy reproducibility files
    copy_code_folder(bids_root, deriv_dir, args.srate, args.highpass)

    # Validate
    validate_bids(deriv_dir)

    print(f'\n=== Done ===')
    print(f'  Derivative: {deriv_dir}')
    print(f'  Code:       {os.path.join(deriv_dir, "code")}')


if __name__ == '__main__':
    main()
