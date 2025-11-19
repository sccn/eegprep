# ---------------------------------------------------------
# Base image
# ---------------------------------------------------------
    #    FROM ubuntu:latest
    FROM --platform=linux/amd64 ubuntu:22.04

    # python:3.10-slim
    # Avoid interactive prompts
    ENV DEBIAN_FRONTEND=noninteractive
    # ---------------------------------------------------------
    # System dependencies
    # ---------------------------------------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libhdf5-dev \
        liblapack-dev \
        libblas-dev \
        libx11-6 \
        && rm -rf /var/lib/apt/lists/*
    # ---------------------------------------------------------
    # Create working directory
    # ---------------------------------------------------------
    WORKDIR /usr/src/eegprep
    # ---------------------------------------------------------
    # Clone EEGPrep from GitHub
    # ---------------------------------------------------------
    RUN git clone https://github.com/sccn/eegprep.git . 
    # ---------------------------------------------------------
    # Install EEGPrep from source
    # Choose ONE of the 2 installation lines below
    # ---------------------------------------------------------
    # Full install including ICLabel (~7GB on Linux)
    # RUN pip install --no-cache-dir ".[all]"
    # Lean install (no ICLabel heavy models)
    RUN pip install --no-cache-dir .
    RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

    # ---------------------------------------------------------
    # Set entrypoint
    # ---------------------------------------------------------
    ENTRYPOINT ["/bin/bash"]