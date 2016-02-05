#!/bin/bash
set -o errexit
set -o xtrace
apt-get update
apt-get install -y \
    curl \
    wget \
    git-core \
    gcc \
    g++ \
    cmake \
    python \
    python2.7 \
    python2.7-dev \
    zlib1g-dev \
    bzip2 \
    libyaml-dev \
    libyaml-0-2
wget https://bootstrap.pypa.io/get-pip.py -O - | python
pip install --upgrade setuptools
pip install wheel
export CC=gcc
export CXX=g++
pip install \
    --cache-dir /usr/local/src/nupic.core/pip-cache \
    --build /usr/local/src/nupic.core/pip-build \
    --no-clean \
    pycapnp==0.5.5 \
    -r bindings/py/requirements.txt
python setup.py bdist bdist_dumb bdist_wheel sdist
pwd
ls -laFh
