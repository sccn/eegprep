# =============================================================
# EEGPrep minimal preprocessing — slim multi-stage Docker image
# Supports: linux/amd64 (Intel/AMD) and linux/arm64 (Apple Silicon, AWS Graviton)
#
# Base image pinned for long-term reproducibility:
#   python:3.12-slim @ sha256:804ddf3251a60bbf9c92e73b7566c40428d54d0e79d3428194edf40da6521286
# =============================================================

# ----- Stage 1: build (has compilers and dev headers) --------
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config \
    libhdf5-dev liblapack-dev libblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy local source and install (no PyTorch, no optional extras)
COPY src/ src/
COPY pyproject.toml LICENSE README.md ./
RUN pip install --no-cache-dir --prefix=/install .

# ----- Stage 2: runtime (no compilers, no dev headers) -------
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Only runtime shared libs needed by numpy/scipy/h5py
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-310 liblapack3 libblas3 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy processing script
COPY scripts/bids_minimal_preproc.py /usr/local/bin/bids_minimal_preproc.py

# Default: process BIDS dataset mounted at /data
# Override with: docker run eegprep-minimal --srate 200 --highpass 1.0
ENTRYPOINT ["python", "/usr/local/bin/bids_minimal_preproc.py", "--input", "/data"]
