#!/bin/bash
set -o errexit
set -o xtrace

# Install pip
python ci/bamboo/get-pip.py --user

# Install python dependencies w/ pip
pip install \
    --upgrade \
    --ignore-installed \
    --user \
    --cache-dir /shared/pip-cache \
    --build /shared/pip-build \
    --no-clean \
    setuptools \
    wheel \
    pycapnp==0.5.8 \
    -r bindings/py/requirements.txt

# Build and install nupic.core
mkdir -p build/scripts
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/build/release -DPY_EXTENSIONS_DIR=`pwd`/bindings/py/nupic/bindings .
make install
./build/release/bin/cpp_region_test
./build/release/bin/unit_tests

# Build wheel
ARCHFLAGS="-arch x86_64" python setup.py bdist_wheel
py.test bindings/py/tests