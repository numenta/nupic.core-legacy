#!/bin/bash
set -o errexit
set -o xtrace

# Fixup $PATH for --user installation
export PATH=${HOME}/Library/Python/2.7/bin:${PATH}

# Install pip
curl --silent --show-error --retry 5 -O http://releases.numenta.org/pip/1ebd3cb7a5a3073058d0c9552ab074bd/get-pip.py
python get-pip.py --user --ignore-installed

# Install python dependencies w/ pip
pip install \
    --upgrade \
    --ignore-installed \
    --user \
    setuptools \
    wheel \
    pycapnp==0.5.8 \
    -r bindings/py/requirements.txt

# Build and install nupic.core
mkdir -p build/scripts
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/build/release -DPY_EXTENSIONS_DIR=`pwd`/bindings/py/nupic/bindings .
make install
./build/release/bin/cpp_region_test
./build/release/bin/py_region_test
./build/release/bin/unit_tests

# Build wheel
ARCHFLAGS="-arch x86_64" python setup.py bdist_wheel
py.test bindings/py/tests
