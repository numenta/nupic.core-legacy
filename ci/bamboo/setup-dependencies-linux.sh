#!/bin/bash

# Environment defaults
if [ -z "${USER}" ]; then
    USER="docker"
fi
export USER

# Setup compiler
if [ -z "${CC}" ]; then
    CC="gcc"
fi
export CC

if [ "${CC}" = "clang" ]; then
    if [ -z "${CXX}" ]; then
        CXX="clang++"
    fi
    COMPILER_PACKAGES="clang-3.4" # Ubuntu-specific apt package name
else
    if [ -z "${CXX}" ]; then
        CXX="g++"
    fi
    COMPILER_PACKAGES="${CC} ${CXX}" # Ubuntu-specific apt package names
fi
export CXX

# Install OS dependencies, assuming stock ubuntu:latest
apt-get update
apt-get install -y \
    curl \
    git-core \
    ${COMPILER_PACKAGES} \
    cmake \
    python \
    python2.7 \
    python2.7-dev

# Install pip
python ci/bamboo/get-pip.py --user
python -m pip install --upgrade --ignore-installed setuptools
python -m pip install wheel
