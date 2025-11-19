FROM --platform=linux/amd64 python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TMPDIR=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    build-essential \
    libhdf5-dev \
    liblapack-dev \
    libblas-dev \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install eegprep from PyPI (lean or full)
# Lean:
RUN python -m pip install --no-cache-dir eegprep
# or full:
# RUN python -m pip install --no-cache-dir "eegprep[all]"

WORKDIR /usr/src/project
ENTRYPOINT ["/bin/bash"]