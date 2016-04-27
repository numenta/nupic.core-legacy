#!/bin/bash
set -o errexit
set -o xtrace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the common setup
${DIR}/setup-dependencies.sh

# Install nupic.core dependencies
pip install \
    --cache-dir /usr/local/src/nupic.core/pip-cache \
    --build /usr/local/src/nupic.core/pip-build \
    --no-clean \
    pycapnp==0.5.5 \
    -r bindings/py/requirements.txt

# Build and install nupic.core
mkdir -p build/scripts
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/build/release -DPY_EXTENSIONS_DIR=`pwd`/bindings/py/nupic/bindings .
make install
./build/release/bin/cpp_region_test
./build/release/bin/unit_tests

# Build installable python packages
python setup.py bdist bdist_dumb bdist_wheel sdist
py.test bindings/py/tests
