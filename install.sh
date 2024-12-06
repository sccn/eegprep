#! /bin/bash

# exit if a command fails
set -e

apt-get update
apt-get install -y software-properties-common
apt-get install -y gcc g++ gfortran make libopenblas-dev liblapack-dev libpcre3-dev libarpack2-dev libcurl4-gnutls-dev epstool libfftw3-dev transfig libfltk1.3-dev libfontconfig1-dev libfreetype6-dev libgl2ps-dev libglpk-dev libreadline-dev gnuplot-x11 libgraphicsmagick++1-dev libhdf5-serial-dev openjdk-8-jdk libsndfile1-dev llvm-dev lpr texinfo libgl1-mesa-dev pstoedit portaudio19-dev libqhull-dev libqrupdate-dev libqscintilla2-dev libsuitesparse-dev texlive texlive-generic-recommended libxft-dev zlib1g-dev autoconf automake bison flex gperf gzip icoutils librsvg2-bin libtool perl rsync tar qtbase5-dev qttools5-dev qttools5-dev-tools libqscintilla2-qt5-dev
apt-get remove -y software-properties-common
apt-get install -y liboctave-dev
apt-get install -y build-essential

# cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# prepare dir
mkdir /source
