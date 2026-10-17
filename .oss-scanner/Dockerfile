# OSS Scanner build for EEGPrep (https://github.com/anthropics/oss-scanner).
# The scanner builds this with the repository root as the build context. The audit and the tests then run with no
# network, so everything they need is fetched here.
FROM debian:trixie
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV DEBIAN_FRONTEND=noninteractive \
    EEGPREP_SKIP_MATLAB=1 \
    QT_QPA_PLATFORM=offscreen \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    MPLCONFIGDIR=/opt/cache/matplotlib \
    MNE_DATA=/opt/cache/mne \
    PATH=/opt/venv/bin:$PATH
# CI's set from .github/workflows/test.yml, plus libraries GitHub's Ubuntu runners already have and Debian lacks.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential pkg-config libsndfile1 libglib2.0-0 \
      libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 \
      libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
      libxcb-shape0 libxcb-xinerama0 \
      libfontconfig1 libfreetype6 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# scanner contract: the checkout lives inside the image, at /src
COPY . /src
WORKDIR /src

# The pinned EEGLAB submodule, without its nested plugin submodules (CI does the same).
RUN if [ ! -f src/eegprep/eeglab/eeglab.m ]; then \
      if [ -e .git ]; then \
        git -c submodule.recurse=false submodule update --init --depth 1 -- src/eegprep/eeglab \
        || git -c submodule.recurse=false submodule update --init -- src/eegprep/eeglab; \
      else \
        echo >&2 'ERROR: src/eegprep/eeglab is empty and there is no .git to fetch the submodule from'; \
        exit 1; \
      fi; \
    fi; \
    test -s src/eegprep/eeglab/eeglab.m \
    || { echo >&2 'ERROR: failed to populate the pinned EEGLAB submodule'; exit 1; }

# Same environment as CI, from uv.lock: all extras (torch is needed by ICLabel) and the dev group.
RUN uv python install 3.12 \
    && uv sync --python 3.12 --all-extras --group dev
# eegprep-lean (packages/eegprep-lean) is a separate distribution with no base dependencies. It gets its own
# environment with both extras, as the last tier in .github/workflows/test.yml does.
RUN uv venv /opt/lean-venv --python 3.12 \
    && uv pip install --python /opt/lean-venv/bin/python -e "packages/eegprep-lean[zarr,plot]" "zarr==3.4.0" pytest
ENV UV_OFFLINE=1
RUN echo 'export PATH=/opt/venv/bin:$PATH' > /etc/profile.d/eegprep-venv.sh

# tests/test_clean_rawdata.py, test_pop_select.py and test_eeg_eegrej.py download FlankerTest.set on first use.
RUN curl -fsSL -o sample_data/FlankerTest.set https://sccntestdatasets.s3.us-east-2.amazonaws.com/FlankerTest.set

# Warm the matplotlib font cache and PyBIDS' packaged configuration.
RUN mkdir -p "$MPLCONFIGDIR" "$MNE_DATA" \
    && python -c 'import matplotlib; matplotlib.use("Agg"); from matplotlib import pyplot as plt; fig = plt.figure(); fig.canvas.draw(); plt.close(fig)' \
    && python -c 'import json, tempfile; from pathlib import Path; from bids import BIDSLayout; from bids.layout import Config; Config.load("bids"); Config.load("derivatives"); t = tempfile.TemporaryDirectory(); p = Path(t.name); (p / "dataset_description.json").write_text(json.dumps({"Name": "offline-warmup", "BIDSVersion": "1.8.0"})); BIDSLayout(p, validate=False); t.cleanup()'

RUN pytest -q tests/test_pop_loadset.py tests/test_pop_loadset_h5.py tests/test_pop_loadbv.py tests/test_extension_catalog.py tests/test_public_api_examples.py
RUN cd packages/eegprep-lean && /opt/lean-venv/bin/python -m pytest -q -p no:cacheprovider tests -m "not network"
CMD ["/bin/bash"]
