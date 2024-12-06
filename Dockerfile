FROM ubuntu:latest
MAINTAINER ome-devel@lists.openmicroscopy.org.uk


ENV DEBIAN_FRONTEND noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python3-dev \
    python3-venv \
    octave \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a Python virtual environment
RUN python3 -m venv /opt/venv

# Ensure pip is up-to-date inside the virtual environment
RUN /opt/venv/bin/pip install --upgrade pip

# Configure pip to use updated PyPI server mirrors
RUN mkdir -p /root/.pip && echo "[global]\nindex-url = https://pypi.org/simple" > /root/.pip/pip.conf

# Install the 'eegrep' package inside the virtual environment
RUN /opt/venv/bin/pip install eegprep

# Set environment path for the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# ADD install.sh install.sh
# RUN sh ./install.sh && rm install.sh

#make it work under singularity
RUN ldconfig && mkdir -p /N/u /N/home /N/dc2 /N/soft

RUN useradd -ms /bin/bash octave
# ADD eeglab /home/octave/eeglab
# ADD load_eeglab.m /home/octave
RUN chown -R octave:octave /home/octave/

USER octave
WORKDIR /home/octave

VOLUME ["/source"]

#https://wiki.ubuntu.com/DashAsBinSh
#RUN rm /bin/sh && ln -s /bin/bash /bin/sh
