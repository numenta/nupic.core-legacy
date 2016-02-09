#!/bin/bash
set -o errexit
set -o xtrace

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
    wget \
    git-core \
    ${COMPILER_PACKAGES} \
    cmake \
    python \
    python2.7 \
    python2.7-dev
wget https://bootstrap.pypa.io/get-pip.py -O - | python
pip install --upgrade --ignore-installed setuptools
pip install wheel

# Install nupic.core dependencies
pip install \
    --cache-dir /usr/local/src/nupic.core/pip-cache \
    --build /usr/local/src/nupic.core/pip-build \
    --no-clean \
    pycapnp==0.5.5 \
    -r bindings/py/requirements.txt

# Build and install nupic.core
mkdir -p build/scripts
cmake . -DNTA_COV_ENABLED=ON -DCMAKE_INSTALL_PREFIX=`pwd`/build/release -DPY_EXTENSIONS_DIR=`pwd`/bindings/py/nupic/bindings
make install
./build/release/bin/cpp_region_test
./build/release/bin/unit_tests

# Build installable python packages
python setup.py bdist bdist_dumb bdist_wheel sdist
py.test bindings/py/tests
